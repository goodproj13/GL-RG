# GL-RG: Global-Local Representation Granularity for Video Captioning

![framework.png](Figs/framework.png)



## Dependencies

* Python 2.7
* Pytorch 0.2
* [Microsoft COCO Caption Evaluation](https://github.com/tylin/coco-caption)
* [CIDEr](https://github.com/plsang/cider)
* torch, numpy, scikit-image, h5py 

This repo was tested with Python 2.7, [PyTorch](https://pytorch.org) 0.2.0, [cuDNN](https://developer.nvidia.com/cudnn) 6.0, and [CUDA](https://developer.nvidia.com/cuda-toolkit) 8.0. But it should be runnable with more recent PyTorch versions.

You can use anaconda or miniconda to install the dependencies:
```bash
conda create GL-RG-pytorch python=2.7 pytorch=0.2 scikit-image h5py
```



## Installation

First clone the this repository to any location using `--recursive`:

```ba
git clone --recursive https://github.com/goodproj13/GL-RG.git
```



Please run following script to download [Stanford CoreNLP 3.6.0](http://stanfordnlp.github.io/CoreNLP/index.html) models to `coco-caption/`:

```bash
cd coco-caption
./get_stanford_models.sh
```

Check out the `coco-caption/`,  `cider/`,  `data/` and `model/` projects into your working directory. If not, please find detailed steps [Here](docs/INSTALL.md) for installation and dataset preparation.



## Model Zoo

| Model | Dataset | Exp. | B@4 | M | R | C | Download Link |
| :--------: | :---------: | :-----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
| GL-RG | MSR-VTT | XE | 45.5  | 30.1 | 62.6 | 51.2 | [Google Drive]() |
| GL-RG | MSR-VTT | DXE | 46.9 | 30.4 | 63.9 | 55.0 | [Google Drive]() |
| GL-RG + IT | MSR-VTT | DR | 46.9 | 31.2 | 65.7 | 60.6 | [Google Drive]() |
| GL-RG | MSVD | XE | 52.3  | 33.8 | 70.4 | 58.7 | [Google Drive]() |
| GL-RG | MSVD | DXE | 57.7 | 38.6 | 74.9 | 95.9 | [Google Drive]() |
| GL-RG + IT | MSVD | DR | 60.5 | 38.9 | 76.4 | 101.0 | [Google Drive]() |



## Test

Check out the trained model weights under the `model/` directory (following [Installation](docs/INSTALL.md) ) and run:
```bash
./test.sh
```

**Note:** Please reset `MODEL_NAME`, `EXP_NAME` and `DATASET` in `test.sh` if running with different models.



## License

`GL-RG` is released under the MIT license.



## Acknowledgements
We are truly thankful of the following prior efforts in terms of knowledge contributions and open-source repos.
+ SA-LSTM: Describing Videos by Exploiting Temporal Structure (ICCV'15) [[paper]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Yao_Describing_Videos_by_ICCV_2015_paper.pdf) [[implement code]](https://github.com/hobincar/SA-LSTM)
+ RecNet: Reconstruction Network for Video Captioning (CVPR'18) [[paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Reconstruction_Network_for_CVPR_2018_paper.pdf) [[official code]](https://github.com/hobincar/RecNet) 
+ SAAT: Syntax-Aware Action Targeting for Video Captioning (CVPR'20) [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zheng_Syntax-Aware_Action_Targeting_for_Video_Captioning_CVPR_2020_paper.pdf) [[official code]](https://github.com/SydCaption/SAAT)
