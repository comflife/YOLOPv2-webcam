# Realtime_yolopv2
A repo where anyone can develop the Yolopv2 algorithm in real time using a webcam. 

Adapted from https://github.com/CAIC-AD/YOLOPv2


## Installation

`conda create -n realtime python=3.8.16`
`conda activate realtime`
`conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
`conda install matplotlib`
`conda install scikit-learn`



## Usage
`cd Realtime_yolopv2`
`python demo_ver2.py --source 0`

## Models

You can get the model from <a href="https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt">here</a>. This is the original YOLOPv2 model and this was trained by BDD100k dataset.


## Resources


### Papers
YOLOPv2: Better, Faster, Stronger for Panoptic Driving Perception

https://arxiv.org/abs/2208.11434
