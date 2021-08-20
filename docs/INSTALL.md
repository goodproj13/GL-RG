## Installation
This repo was tested with Python 2.7, [PyTorch](https://pytorch.org) 0.2.0, [cuDNN](https://developer.nvidia.com/cudnn) 6.0, and [CUDA](https://developer.nvidia.com/cuda-toolkit) 8.0. But it should be runnable with recent PyTorch versions >=0.2.0. (0.3.x may be also ok)

You can use anaconda or miniconda to install the dependencies:
```bash
conda create GL-RG-pytorch python=2.7 pytorch=0.2 scikit-image h5py
```

### Microsoft COCO Caption Evaluation

Please run following script to download evaluation codes for [Microsoft COCO Caption Evaluation](https://github.com/tylin/coco-caption) and put them under the `coco-caption` directory:

```ba
cd GL-RG
git clone https://github.com/tylin/coco-caption.git
```

Please refer to [Microsoft COCO Caption Evaluation](https://github.com/tylin/coco-caption) for more setup details.

### CIDEr Code

Please run following script to download [Consensus-based Image Description Evaluation](https://github.com/plsang/cider) and put them under the `cider` directory:

```ba
cd GL-RG
git clone https://github.com/plsang/cider.git
```



## Preparation

### Datasets

Download the `metadate` and `feature` from [Google Drive]() and then unzip them under the `data` directory like:
```shell
data
├── feature
│   ├── msrvtt_test_sem_tag_res.h5
│   ├── msrvtt_train_sem_tag_res.h5
│   ├── msrvtt_val_sem_tag_res.h5
│   ├── msvd_test_sem_tag_res.h5
│   ├── msvd_train_sem_tag_res.h5
│   └── msvd_val_sem_tag_res.h5
│
├── metadata
│   ├── msrvtt_test_cocofmt.json
│   ├── msrvtt_test_sequencelabel.h5
│   ├── msvd_test_cocofmt.json
└── └── msvd_test_sequencelabel.h5
```

### Pre-trained Weights

Download the pre-trained weights from [Google Drive]() and then unzip them under the `model` directory like:
```shell
model
├── GL-RG_XE_msrvtt
│   └── model.pth
│
├── GL-RG_XE_msvd
│   └── model.pth
│
├── GL-RG_DXE_msrvtt
│   └── model.pth
│
├── GL-RG_DXE_msvd
│   └── model.pth
│
├── GL-RG_DR_msrvtt
│   └── model.pth
│
└── GL-RG_DR_msvd
    └── model.pth
```

