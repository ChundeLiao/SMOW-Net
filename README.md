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
