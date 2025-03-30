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

# 💬 Training and Testing
train.py

test.py
# 💬 Model Weights
Find the model weights [here](https://drive.google.com/drive/folders/1GKj99WhwkV6j2tNnrAqvb0etJPY0N-Hm?usp=sharing)
# 💬 License
The code is released for non-commercial and research purposes only. For commercial purposes, please contact the authors.
