# YOLOPv3
A repository for developing the YOLOPv2 model in real time using a webcam.
Added Bev and more advanced algorithms.

Adapted from https://github.com/CAIC-AD/YOLOPv2

## Info
This code is developed to detect lane & drivable area in Autonomous driving competition. Made by Byounggun Park(Comflife).

## Installation
(check your cuda, pytorch version before install)

```bash
conda create -n realtime python=3.8.16
conda activate realtime
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install matplotlib
conda install scikit-learn
pip install opencv-python
```


## Usage
(check your camera number --source 0,1,2)
```bash
python3 realtime.py --source 0
```


## Models

You can get the model from <a href="https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt">here</a>. This is the original YOLOPv2 model and this was trained by BDD100k dataset.


## Resources


### Papers
YOLOPv2: Better, Faster, Stronger for Panoptic Driving Perception

https://arxiv.org/abs/2208.11434
