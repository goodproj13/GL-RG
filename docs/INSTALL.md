## Installation
This repo was tested with Python 2.7, [PyTorch](https://pytorch.org) 0.2.0, [cuDNN](https://developer.nvidia.com/cudnn) 6.0, and [CUDA](https://developer.nvidia.com/cuda-toolkit) 8.0. But it should be runnable with recent PyTorch versions >=0.2.0. (0.3.x may be also ok)

You can use anaconda or miniconda to install the dependencies:
```bash
conda create GL-RG-pytorch python=2.7 pytorch=0.2 scikit-image h5py
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

