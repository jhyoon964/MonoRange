# MonoRange: Monocular 3D Object Detection Based on Object-centric RangeMap in Adverse Weather Conditions

## Getting Started
### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA CuDNN (Not mandatory bur recommended)
- Python 3

### Installation
- Dependencies:  
	1. lpips
	2. wandb
	3. pytorch
	4. torchvision
	5. matplotlib
	6. dlib
- All dependencies can be installed using *pip install* and the package name

### Running MonoRange
The primary training script is `train.sh`. It takes aligned and cropped images from the paths specified in the "Input info" subsection of `configs/paths_config.py`.
The results, including inversion latent codes and optimized generators, are saved to the directories listed under "Dirs for output files" in `configs/paths_config.py`.
