import torch
import torch.nn.functional as F
from torch import nn
import math
import copy

from utils import box_ops
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                            accuracy, get_world_size, interpolate,
                            is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .det2d_transformer import build_det2d_transformer
from .det3d_transformer import build_det3d_transformer

from .region_seg_head import RegionSegHead
from .rangemap import rangemap
from .rangemap.ddn_loss import DDNLoss
from lib.losses.focal_loss import sigmoid_focal_loss
from .position_encoding import PositionEmbeddingCamRay


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class monorange(nn.Module):
    """ This is the monorange module that performs monocualr 3D object detection """
    def __init__(self, backbone, rangemap, det2d_transformer, det3d_transformer,
                  num_classes, num_queries, num_feature_levels, 
                  aux_loss=True, with_box_refine=False, init_box=False, group_num=11):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            det2d_transformer: transformer architecture. See det2d_transformer.py
            det3d_transformer: transformer architecture. See det3d_transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For KITTI, we recommend 50 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
        """
        super().__init__()
 
        self.num_queries = num_queries
        self.det2d_transformer = det2d_transformer
        self.det3d_transformer = det3d_transformer
        self.rangemap = rangemap
        hidden_dim = det2d_transformer.d_model
        self.hidden_dim = hidden_dim
        
        self.region_head = RegionSegHead(d_model=hidden_dim)

        self.num_feature_levels = num_feature_levels
        
        # prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
        self.dim_embed_3d = MLP(hidden_dim, hidden_dim, 3, 2)
        self.angle_embed = MLP(hidden_dim, hidden_dim, 24, 2)
        self.range_embed = MLP(hidden_dim, hidden_dim, 2, 2)  # range and deviation

        if init_box == True:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        self.query_embed = nn.Embedding(num_queries * group_num, hidden_dim*2)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.num_classes = num_classes

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        num_pred = det3d_transformer.decoder.num_layers + 1
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.det2d_transformer.decoder.bbox_embed = self.bbox_embed
            self.det3d_transformer.decoder.bbox_embed = self.bbox_embed
            
            self.dim_embed_3d = _get_clones(self.dim_embed_3d, num_pred)
            self.angle_embed = _get_clones(self.angle_embed, num_pred)
            self.range_embed = _get_clones(self.range_embed, num_pred)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.dim_embed_3d = nn.ModuleList([self.dim_embed_3d for _ in range(num_pred)])
            self.angle_embed = nn.ModuleList([self.angle_embed for _ in range(num_pred)])
            self.range_embed = nn.ModuleList([self.range_embed for _ in range(num_pred)])
            self.rangeaware_transformer.decoder.bbox_embed = None


    def forward(self, images, calibs, targets, img_sizes, dn_args=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """

        features, pos = self.backbone(images)
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = torch.zeros(src.shape[0], src.shape[2], src.shape[3]).to(torch.bool).to(src.device)
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        # region enhancement
        enhanced_srcs, region_probs, seg_embed = self.region_head(srcs)

        if self.training:
            query_embeds = self.query_embed.weight
        else:
            # only use one group in inference
            query_embeds = self.query_embed.weight[:self.num_queries]

        srcs = enhanced_srcs
        pred_range_map_logits, range_pos_embed, weighted_range = self.rangemap(srcs, masks[1], seg_embed[1] + pos[1])
        
        #pos_3d = []
        # for l, feat in enumerate(features):
        #     range_pos_3d = self.position_embed(feat, calibs=None, range_map = pred_range_map_logits)
        #     pos[l] = range_pos_3d
        #pos = pos_3d

        intermediate_output = self.det2d_transformer(srcs, masks, pos, query_embeds)
        
        hs_2d = intermediate_output['hs']
        init_reference_2d = intermediate_output['init_reference_out']
        inter_references_2d = intermediate_output['inter_references_out']
        
        inter_coords = []
        inter_classes = []

        for lvl in range(hs_2d.shape[0]):
            if lvl == 0:
                reference = init_reference_2d
            else:
                reference = inter_references_2d[lvl - 1]
            reference = inverse_sigmoid(reference)

            tmp = self.bbox_embed[lvl](hs_2d[lvl])
            if reference.shape[-1] == 6:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            # 3d center + 2d box
            inter_coord = tmp.sigmoid()
            inter_coords.append(inter_coord)

            # classes
            inter_class = self.class_embed[lvl](hs_2d[lvl])
            inter_classes.append(inter_class)

        inter_coord = torch.stack(inter_coords)
        inter_class = torch.stack(inter_classes)

        query_embeds = hs_2d[-1]
        hs, init_reference, inter_references = self.det3d_transformer(intermediate_output, query_embeds, range_pos_embed)

        outputs_coords = []
        outputs_classes = []
        outputs_3d_dims = []       
        outputs_ranges = []
        outputs_angles = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 6:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            # 3d center + 2d box
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)

            # classes
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_classes.append(outputs_class)

            # 3D sizes
            size3d = self.dim_embed_3d[lvl](hs[lvl])
            outputs_3d_dims.append(size3d)

            # range_geo_err
            range_geo_err = self.range_embed[lvl](hs[lvl])
            
            # range_geo
            box2d_height_norm = outputs_coord[:, :, 4] + outputs_coord[:, :, 5]
            box2d_height = torch.clamp(box2d_height_norm * img_sizes[:, 1: 2], min=1.0)
            range_geo = size3d[:, :, 0]/ box2d_height * calibs[:, 0, 0].unsqueeze(1)
            
            # range_map
            # outputs_center3d = ((outputs_coord[..., :2] - 0.5) * 2).unsqueeze(2)   #.detach()
            # range_map = F.grid_sample(
            #     weighted_range.unsqueeze(1),
            #     outputs_center3d,
            #     mode='bilinear',
            #     align_corners=True).squeeze(1)    
            
            # range average + sigma
            # range_ave = torch.cat([( (1. / (range_reg[:, :, 0: 1].sigmoid() + 1e-6) - 1.) + range_geo.unsqueeze(-1) + range_map) / 3,
            
            range_ave = torch.cat([range_geo.unsqueeze(-1) + range_geo_err[:, :, 0: 1],          
                                    range_geo_err[:, :, 1: 2]], -1)

            outputs_ranges.append(range_ave)

            # angles
            outputs_angle = self.angle_embed[lvl](hs[lvl])
            outputs_angles.append(outputs_angle)

        outputs_coord = torch.stack(outputs_coords)
        outputs_class = torch.stack(outputs_classes)
        outputs_3d_dim = torch.stack(outputs_3d_dims)
        outputs_range = torch.stack(outputs_ranges) 
        outputs_angle = torch.stack(outputs_angles)
  
        #########################
        projected_2d_boxes = self.project_3d_to_2d(outputs_coord, calibs)
  
        out = dict()
        out['pred_logits'] = outputs_class[-1]
        out['pred_boxes'] = outputs_coord[-1]
        out['pred_3d_dim'] = outputs_3d_dim[-1]
        out['pred_range'] = outputs_range[-1]
        out['pred_angle'] = outputs_angle[-1]
        out['pred_range_map_logits'] = pred_range_map_logits
        out['pred_region_prob'] = region_probs
        
        
        ##########################
        out['projected_2d_boxes'] = projected_2d_boxes  # New output for cycle consistency


        out['inter_outputs'] = self._set_inter_loss(inter_class, inter_coord)
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_range) 
        
        return out
    
    
    def project_3d_to_2d(self, outputs_coord, calibs):
        """Project 3D box coordinates to 2D box coordinates."""
        # print(f"Debug: outputs_coord shape: {outputs_coord.shape}")
        outputs_coord = outputs_coord[-1]  # Shape: (batch_size, num_queries, 6)

        batch_size, num_queries, _ = outputs_coord.shape
        projected_2d = []
        for i in range(batch_size):
            calib = calibs[i]  # Camera calibration matrix
            coords_3d = outputs_coord[i, :, :3]  # Extract 3D coordinates
            coords_3d_homo = torch.cat([coords_3d, torch.ones_like(coords_3d[:, :1])], dim=-1)
            coords_2d = torch.matmul(calib, coords_3d_homo.T).T  # Apply projection
            coords_2d = coords_2d[:, :2] / coords_2d[:, 2:3]  # Normalize by range
            projected_2d.append(coords_2d)
        return torch.stack(projected_2d, dim=0)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_range):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 
                 'pred_3d_dim': c, 'pred_angle': d, 'pred_range': e}
                for a, b, c, d, e in zip(outputs_class[:-1], outputs_coord[:-1],
                                         outputs_3d_dim[:-1], outputs_angle[:-1], outputs_range[:-1])]
    
    @torch.jit.unused
    def _set_inter_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class, outputs_coord)]


class SetCriterion(nn.Module):
    """ This class computes the loss for monorange.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses, inter_losses, group_num=11):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.inter_losses = inter_losses
        self.focal_alpha = focal_alpha
        self.ddn_loss = DDNLoss()  # for range map
        self.bce = nn.BCELoss()
        self.bce_noReduce = nn.BCELoss(reduction='none')

        self.group_num = group_num

    ########################################
    def loss_cycle_consistency(self, outputs, targets, indices, num_boxes):
        """Compute cycle consistency loss."""
        idx = self._get_src_permutation_idx(indices)
        pred_2d_boxes = outputs['pred_boxes'][:, :, 2:6][idx]
        projected_2d_boxes = outputs['projected_2d_boxes'][idx]

        # L1 loss between predicted 2D boxes and projected 2D boxes
        loss_cycle = F.l1_loss(pred_2d_boxes, projected_2d_boxes, reduction='none')
        losses = {'loss_cycle_consistency': loss_cycle.sum() / num_boxes}
        return losses
    
    
    
    
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o.squeeze().long()

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_3dcenter(self, outputs, targets, indices, num_boxes):
        
        idx = self._get_src_permutation_idx(indices)
        src_3dcenter = outputs['pred_boxes'][:, :, 0: 2][idx]
        target_3dcenter = torch.cat([t['boxes_3d'][:, 0: 2][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_3dcenter = F.l1_loss(src_3dcenter, target_3dcenter, reduction='none')
        losses = {}
        losses['loss_center'] = loss_3dcenter.sum() / num_boxes
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_2dboxes = outputs['pred_boxes'][:, :, 2: 6][idx]
        target_2dboxes = torch.cat([t['boxes_3d'][:, 2: 6][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # l1
        loss_bbox = F.l1_loss(src_2dboxes, target_2dboxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # giou
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcylrtb_to_xyxy(src_boxes),
            box_ops.box_cxcylrtb_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_ranges(self, outputs, targets, indices, num_boxes):  

        idx = self._get_src_permutation_idx(indices)
   
        src_ranges = outputs['pred_range'][idx]
        target_ranges = torch.cat([t['range'][i] for t, (_, i) in zip(targets, indices)], dim=0).squeeze()
         
        range_loss = 0
        range_input, range_log_variance = src_ranges[:, 0], src_ranges[:, 1] 
        range_loss += 1.4142 * torch.exp(-range_log_variance) * torch.abs(range_input - target_ranges) + range_log_variance
        
        losses = {}
        losses['loss_range'] = range_loss.sum() / num_boxes 
        return losses  
    
    def loss_dims(self, outputs, targets, indices, num_boxes):  

        idx = self._get_src_permutation_idx(indices)
        src_dims = outputs['pred_3d_dim'][idx]
        target_dims = torch.cat([t['size_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        dimension = target_dims.clone().detach()
        dim_loss = torch.abs(src_dims - target_dims)
        dim_loss /= dimension
        with torch.no_grad():
            compensation_weight = F.l1_loss(src_dims, target_dims) / dim_loss.mean()
        dim_loss *= compensation_weight
        losses = {}
        losses['loss_dim'] = dim_loss.sum() / num_boxes
        return losses

    def loss_angles(self, outputs, targets, indices, num_boxes):  

        idx = self._get_src_permutation_idx(indices)
        heading_input = outputs['pred_angle'][idx]
        target_heading_cls = torch.cat([t['heading_bin'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_heading_res = torch.cat([t['heading_res'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        heading_input = heading_input.view(-1, 24)
        heading_target_cls = target_heading_cls.view(-1).long()
        heading_target_res = target_heading_res.view(-1)

        # classification loss
        heading_input_cls = heading_input[:, 0:12]
        cls_loss = F.cross_entropy(heading_input_cls, heading_target_cls, reduction='none')

        # regression loss
        heading_input_res = heading_input[:, 12:24]
        cls_onehot = torch.zeros(heading_target_cls.shape[0], 12).cuda().scatter_(dim=1, index=heading_target_cls.view(-1, 1), value=1)
        heading_input_res = torch.sum(heading_input_res * cls_onehot, 1)
        reg_loss = F.l1_loss(heading_input_res, heading_target_res, reduction='none')
        
        angle_loss = cls_loss + reg_loss
        losses = {}
        losses['loss_angle'] = angle_loss.sum() / num_boxes 
        return losses

    def loss_range_map(self, outputs, targets, indices, num_boxes):
        range_map_logits = outputs['pred_range_map_logits']

        num_gt_per_img = [len(t['boxes']) for t in targets]
        gt_boxes2d = torch.cat([t['boxes'] for t in targets], dim=0) * torch.tensor([80, 24, 80, 24], device='cuda')
        gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)
        gt_center_range = torch.cat([t['range'] for t in targets], dim=0).squeeze(dim=1)
        
        losses = dict()

        losses["loss_range_map"] = self.ddn_loss(
            range_map_logits, gt_boxes2d, num_gt_per_img, gt_center_range)
        return losses

    def loss_region(self, outputs, targets, indices, num_boxes):
        region_probs = outputs['pred_region_prob']
        gt_region = torch.cat([t['obj_region'].unsqueeze(0) for t in targets], dim=0)

        loss = 0
        losses = dict()
        for region_prob in region_probs:
            gt_region_resized = F.interpolate(gt_region.unsqueeze(1).float(), size=region_prob.shape[2:], mode='bilinear', align_corners=True)
            # Compute intersection and union
            intersection = (region_prob * gt_region_resized).sum()
            total = region_prob.sum() + gt_region_resized.sum()
            # Compute Dice Coefficient
            dice_coef = (2. * intersection + 1) / (total + 1)
            # Compute Dice Loss
            dice_loss = 1 - dice_coef
            loss += dice_loss

        losses['loss_region'] = loss

        return losses
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'ranges': self.loss_ranges,
            'dims': self.loss_dims,
            'angles': self.loss_angles,
            'center': self.loss_3dcenter,
            'range_map': self.loss_range_map,
            'region': self.loss_region,
            'cycle_consistency': self.loss_cycle_consistency,
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, mask_dict=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        group_num = self.group_num if self.training else 1
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets) * group_num
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        losses = {}

        # Compute Det 2D loss
        for i, inter_outputs in enumerate(outputs['inter_outputs']):
            indices = self.matcher(inter_outputs, targets, group_num=group_num)
            for loss in self.inter_losses:
                l_dict = self.get_loss(loss, inter_outputs, targets, indices, num_boxes)
                l_dict = {k + f'_inter_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
        
        # Compute Det 2D and 3D loss
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'inter_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, group_num=group_num)
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, group_num=group_num)
                for loss in self.losses:
                    if loss == 'range_map' or loss == 'region':
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(cfg):
    # backbone
    backbone = build_backbone(cfg)

    # detr
    det2d_transformer = build_det2d_transformer(cfg)
    det3d_transformer = build_det3d_transformer(cfg)

    # range prediction module
    rangemap = rangePredictor(cfg)

    model = monorange(
        backbone = backbone,
        rangemap = rangemap,
        det2d_transformer = det2d_transformer,
        det3d_transformer = det3d_transformer,
        num_classes=cfg['num_classes'],
        num_queries=cfg['num_queries'],
        aux_loss=cfg['aux_loss'],
        num_feature_levels=cfg['num_feature_levels'],
        with_box_refine=cfg['with_box_refine'],
        init_box=cfg['init_box'],
        group_num=cfg['group_num'])

    # matcher
    matcher = build_matcher(cfg)

    # loss
    weight_dict = {'loss_ce': cfg['cls_loss_coef'], 'loss_bbox': cfg['bbox_loss_coef']}
    weight_dict['loss_giou'] = cfg['giou_loss_coef']
    weight_dict['loss_dim'] = cfg['dim_loss_coef']
    weight_dict['loss_angle'] = cfg['angle_loss_coef']
    weight_dict['loss_range'] = cfg['range_loss_coef']
    weight_dict['loss_center'] = cfg['3dcenter_loss_coef']
    weight_dict['loss_range_map'] = cfg['range_map_loss_coef']
    weight_dict['loss_region'] = cfg['region_loss_coef']
    
    weight_dict['loss_cycle_consistency'] = cfg['cycle_loss_coef']

    
    if cfg['aux_loss']:
        aux_weight_dict = {}
        for i in range(cfg['dec_layers'] - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    inter_weight_dict = {}
    inter_keys = ['loss_ce', 'loss_bbox', 'loss_center', 'loss_giou']
    layers = cfg['dec_layers']
    for i in range(layers):
        inter_weight_dict.update({k + f'_inter_{i}': v for k, v in weight_dict.items() if k in inter_keys})
    weight_dict.update(inter_weight_dict)
        
    inter_losses = ['labels', 'boxes', 'center']
    losses = ['labels', 'boxes', 'cardinality', 'ranges', 'dims', 'angles', 'center', 'range_map', 'region']

    criterion = SetCriterion(
        cfg['num_classes'],
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=cfg['focal_alpha'],
        losses=losses,
        inter_losses=inter_losses,
        group_num=cfg['group_num']
        )

    device = torch.device(cfg['device'])
    criterion.to(device)
    
    return model, criterion
