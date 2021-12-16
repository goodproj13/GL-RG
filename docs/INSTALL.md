## Installation

**Important:** If you use `--recursive` to clone this repository, skip this section.

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

If you have cloned this repository from [this GitHub](https://github.com/goodproj13/GL-RG), skip this section.

### Datasets

Download the `metadate` and `feature` from [Here](https://github.com/goodproj13/GL-RG/tree/main/data) and then unzip them under the `data` directory like:
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

Download the pre-trained weights from [Here](https://github.com/goodproj13/GL-RG/tree/main/model) and then unzip them under the `model` directory like:
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

