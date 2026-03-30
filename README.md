# LumiGS-SLAM: Radiometrically Robust 3D Gaussian Splatting SLAM for Mobile Mapping in Dynamic Environments

## Installation

Please follow the instructions below to install the repo and dependencies.

```bash
git clone https://github.com/lumigs-slam/LumiGS-SLAM.git
cd LumiGS-SLAM
```
### Install the environment

```bash
# Create conda environment
conda create -n LumiGS-SLAM python=3.10
conda activate LumiGS-SLAM

# Install the requirements
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt

# Build extension 
cd diff-gaussian-rasterization-w-depth
python setup.py install

```

## Dataset

We use [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) and [KITTI](https://www.cvlibs.net/datasets/kitti/) datasets.

For the training of the optimal weights of GRU, we use the [TartanAir](https://theairlab.org/tartanair-dataset).

## Run

Before running `LumiGS-SLAM`, you need to run the data preprocessing scripts first to generate depth images and global features.

1. Data preprocessing

- EuRoC: run `tools/euroc_parser/operate_euroc_data.py`
- KITTI: run `tools/kitti_parser/operate_kitti_data.py`

2. Frontend + Loop Closure

Run `scripts/loop_closure.py`:

```bash
python scripts/loop_closure.py configs/euroc/lumigsslam.py
```

3. Backend Optimization (Pose Graph + Structure Refinement)

Before running `tools/loop_closure/pose_graph_part_optim.py`, you need to run `scripts/slam.py`:

```bash
python scripts/slam.py configs/euroc/lumigsslam.py
```

Then run the backend script (pose graph and structure refinement):

```bash
python tools/loop_closure/pose_graph_part_optim.py
```
## Acknowledgement

Our codebase builds on the code in [LSG-SLAM](https://github.com/lsg-slam/LSG-SLAM.git),
[SplaTAM](https://github.com/spla-tam/SplaTAM.git),
[Luminance-GS](https://github.com/cuiziteng/Luminance-GS.git).
