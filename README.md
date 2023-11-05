# LETNet
This repository is an official PyTorch implementation of our paper "Lightweight Real-time Semantic Segmentation
Network with Efficient Transformer and CNN". Accepted by IEEE TRANSACTIONS ON INTELLIGENCE TRANSPORTATION SYSTEMS, 2023. (IF: 9.551)

[Paper](https://arxiv.org/abs/2302.10484) | [Code](https://github.com/XU-GITHUB-curry/LETNet_Lightweight-Real-time-Semantic-Segmentation-Network-with-Efficient-Transformer-and-CNN)

## Installation

```
cuda == 10.2
Python == 3.6.4
Pytorch == 1.8.0+cu101

# clone this repository
git clone https://github.com/XU-GITHUB-curry/LETNet_Lightweight-Real-time-Semantic-Segmentation-Network-with-Efficient-Transformer-and-CNN.git

```

## Train

```
# cityscapes
python train.py --dataset cityscapes --train_type train --max_epochs 1000 --lr 4.5e-2 --batch_size 5

# camvid
python train.py --dataset cityscapes --train_type train --max_epochs 1000 --lr 1e-3 --batch_size 8
```



## Test

```
# cityscapes
python test.py --dataset cityscapes --checkpoint ./checkpoint/cityscapes/FBSNetbs4gpu1_train/model_1000.pth

# camvid
python test.py --dataset camvid --checkpoint ./checkpoint/camvid/FBSNetbs6gpu1_trainval/model_1000.pth
```

## Predict
only for cityscapes dataset
```
python predict.py --dataset cityscapes 
```

## Results

- Please refer to our article for more details.

| Methods |  Dataset   | Input Size | mIoU(%) |
| :-----: | :--------: | :--------: | :-----: |
| LETNet  | Cityscapes |  512x1024  |  72.8   |
| LETNet  |   CamVid   |  360x480   |  70.5   |



## Citation

If you find this project useful for your research, please cite our paper:

```
@article{xu2023lightweight,
  title={Lightweight Real-Time Semantic Segmentation Network With Efficient Transformer and CNN},
  author={Xu, Guoan and Li, Juncheng and Gao, Guangwei and Lu, Huimin and Yang, Jian and Yue, Dong},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2023},
  publisher={IEEE}
}
```
## Thanks && Refer

```bash
@misc{Efficient-Segmentation-Networks,
  author = {Yu Wang},
  title = {Efficient-Segmentation-Networks Pytorch Implementation},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks}},
  commit = {master}
}
```
For more code about lightweight real-time semantic segmentation, please refer to: https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks

