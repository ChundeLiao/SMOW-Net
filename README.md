# 💬 Requirements
```
Python 3.9.21
pytorch 1.13.1
torchvision 0.14.1
albumentations 1.3.1
einops 0.8.1
numpy 1.26.4
opencv-python 4.11.0.86
timm 1.0.15
tqdm 4.67.1

Please see `requirements.txt` for all the other requirements.
```
# 💬 Dataset Preparation
## 👉 Data Structure
```
"""
Change detection data set with pixel-level binary labels;
├─A
├─B
├─label
└─list
  ├─train.txt
  ├─val.txt
  ├─test.txt
"""
```
`A`: Images of T1 time;

`B`: Images of T2 time;

`label`: label maps;

`list`: contrains `train.txt`, `val.txt`, and `test.txt`. each fild records the name of image paris (XXX.png).
## 👉 Data Download
GVLM-CD: [Google Drive](https://drive.google.com/file/d/1jqcY0U4pl4UR1DKN2rs_R3WAYGy6ISEY/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1B7rBkQNt4C7hUDMtrXXzLg?pwd=vcgd)

LEVIR-CD: [Google Drive](https://drive.google.com/file/d/1_q3UjW5NAgQe05Lg_wWf4cvxpwdT4pRA/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1B0KaEaZ1g1rY6IoL1svjcw?pwd=tfkh)

WHU-CD: [Google Drive](https://drive.google.com/file/d/1owVmai-WK7nSl4E_ahvBSDnW1esljjkN/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1aamE0IOv-yrrH-uDaKshcQ?pwd=xhbq)
# 💬 Training and Testing
train.py

test.py
# 💬 Model Weights
model weights: [Google Drive](https://drive.google.com/drive/folders/1GKj99WhwkV6j2tNnrAqvb0etJPY0N-Hm?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1O7A02m03anLtfxxGyIfgIQ?pwd=164w)
# 💬 License
The code is released for non-commercial and research purposes only. For commercial purposes, please contact the authors.
# 💬 Cite
```
@article{LIAO2026113855,
title = {Remote sensing change detection via spatiotemporal multi-scale fusion and optical flow warping},
journal = {Pattern Recognition},
volume = {179},
pages = {113855},
year = {2026},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2026.113855},
url = {https://www.sciencedirect.com/science/article/pii/S0031320326008204},
author = {Chunde Liao and Kuangrong Hao and Bing Wei and Xue-song Tang and Lihong Ren},
keywords = {Deep learning, Remote sensing, Change detection, Spatiotemporal multi-scale fusion, Optical flow warping},
abstract = {Remote sensing (RS) images change detection (CD) is essential for the surveillance and prevention of geohazards. Nevertheless, the current deep learning (DL)-based CD methods still face challenges such as pseudo changes, missed detections, and edge noise due to the inadequate research of temporal differences and inconsistent viewing angles between the dual-temporal images. In order to improve the perception of spatiotemporal variations and effectively manage complex motion in spatiotemporal data, this paper proposes a spatiotemporal multi-scale fusion and optical flow warping network (SMOW-Net). Initially, the internal fusion property of 3D convolution enables the simultaneous extraction and fusion of feature information in dual-temporal images. The spatiotemporal multi-scale feature encoder (SMFE) module is proposed to mitigate the semantic gap between low-level and high-level features. This module is designed to aggregate complementary feature information between each level through temporal and spatial independent processing and flexible temporal transposed convolutional layers. Furthermore, the optical flow warper (OFW) module is intended to improve the spatiotemporal dynamic modeling capability in order to manage complex motion data effectively, where a two-channel spatial deformation field is autonomously learned by the network to guide feature alignment. The performance advantage of our network over eleven state-of-the-art methods (SOTA) on the GVLM-CD, LEVIR-CD, WHU-CD, S2Looking, and LEVIR-CD+ datasets is validated by experimental results. Finally, we also introduce SMOW-Net-LW, a lightweight variant with significantly reduced model complexity, suitable for resource-constrained settings, while still achieving excellent performance. The code for this work is available at https://github.com/ChundeLiao/SMOW-Net.}
}
```
