import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TransformerEncoder, TransformerEncoderLayer


class rangePredictor(nn.Module):

    def __init__(self, model_cfg):
        """
        Initialize range predictor and range encoder
        Args:
            model_cfg [EasyDict]: range classification network config
        """
        super().__init__()
        range_num_bins = int(model_cfg["num_range_bins"])
        range_min = float(model_cfg["range_min"])
        range_max = float(model_cfg["range_max"])
        self.range_max = range_max

        bin_size = 2 * (range_max - range_min) / (range_num_bins * (1 + range_num_bins))
        bin_indice = torch.linspace(0, range_num_bins - 1, range_num_bins)
        bin_value = (bin_indice + 0.5).pow(2) * bin_size / 2 - bin_size / 8 + range_min
        bin_value = torch.cat([bin_value, torch.tensor([range_max])], dim=0)
        self.range_bin_values = nn.Parameter(bin_value, requires_grad=False)

        # Create modules
        d_model = model_cfg["hidden_dim"]
        self.downsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GroupNorm(32, d_model))
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model))
        self.upsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model))

        self.range_head = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU())

        self.range_classifier = nn.Conv2d(d_model, range_num_bins + 1, kernel_size=(1, 1))

        range_encoder_layer = TransformerEncoderLayer(
            d_model, nhead=8, dim_feedforward=256, dropout=0.1)
        self.range_encoder = TransformerEncoder(range_encoder_layer, 1)
        
        self.range_pos_embed = nn.Embedding(int(self.range_max) + 1, d_model)


    def forward(self, feature, mask, pos):

        # foreground range map
        src_16 = self.proj(feature[1])
        src_32 = self.upsample(F.interpolate(feature[2], size=src_16.shape[-2:], mode='bilinear'))
        src_8 = self.downsample(feature[0])

        src = (src_8 + src_16 + src_32) / 3

        src = self.range_head(src)
        range_logits = self.range_classifier(src)
        
        # Calculate the median value along the range_num_bins dimension
        #median_values = torch.median(range_logits, dim=1, keepdim=True).values
        # Apply the threshold: set values below the median to zero
        #thresholded_logits = torch.where(range_logits >= median_values, range_logits,  torch.tensor(-float('inf')).to(range_logits.device))
             
        range_probs = F.softmax(range_logits, dim=1)
        weighted_range = (range_probs * self.range_bin_values.reshape(1, -1, 1, 1)).sum(dim=1)
        
        # range embeddings with range positional encodings
        B, C, H, W = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        pos = pos.flatten(2).permute(2, 0, 1)

        range_embed = self.range_encoder(src, mask, pos)
        range_embed = range_embed.permute(1, 2, 0).reshape(B, C, H, W)
        
        range_pos_embed_ip = self.interpolate_range_embed(weighted_range)
        range_embed = range_embed #+ range_pos_embed_ip
        
        return range_logits, range_embed, weighted_range


    def interpolate_range_embed(self, range):
        range = range.clamp(min=0, max=self.range_max)
        pos = self.interpolate_1d(range, self.range_pos_embed)
        pos = pos.permute(0, 3, 1, 2)
        return pos

    def interpolate_1d(self, coord, embed):
        floor_coord = coord.floor()
        delta = (coord - floor_coord).unsqueeze(-1)
        floor_coord = floor_coord.long()
        ceil_coord = (floor_coord + 1).clamp(max=embed.num_embeddings - 1)
        return embed(floor_coord) * (1 - delta) + embed(ceil_coord) * delta
