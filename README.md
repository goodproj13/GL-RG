# GL-RG: Global-Local Representation Granularity for Video Captioning

![framework.png](Figs/framework.png)

`GL-RG` exploit extensive vision representations from different video ranges to improve linguistic expression. We devise a novel global-local encoder to produce rich semantic vocabulary. With our incremental training strategy, `GL-RG` successfully leverages the global-local vision representation to achieve fine-grained captioning on video contents. 



## Dependencies

* Python 2.7
* Pytorch 0.2 or 1.0
* Microsoft COCO Caption Evaluation
* CIDEr
* numpy, scikit-image, h5py, requests 

This repo was tested with Python 2.7, PyTorch 0.2.0 (1.0.1), cuDNN 6.0 (10.0), and CUDA 8.0. But it should be runnable with more recent PyTorch>=1.0 (or >=0.2, <=1.0) versions.

You can use anaconda or miniconda to install the dependencies:
```bash
conda create -n GL-RG-pytorch python=2.7 pytorch=0.2 scikit-image h5py requests
```



## Installation


Check out the `coco-caption/`,  `cider/`,  `data/` and `model/` projects into your working directory. If not, please find detailed steps [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.

Please run following script to download Stanford CoreNLP 3.6.0 models to `coco-caption/`:

```bash
cd coco-caption
./get_stanford_models.sh
```



## Model Zoo

| Model | Dataset | Exp. | B@4 | M | R | C | Download Link |
| :--------: | :---------: | :-----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
| GL-RG | MSR-VTT | XE | 45.5  | 30.1 | 62.6 | 51.2 | [GL-RG_XE_msrvtt](model/GL-RG_XE_msrvtt) |
| GL-RG | MSR-VTT | DXE | **46.9** | 30.4 | 63.9 | 55.0 | [GL-RG_DXE_msrvtt](model/GL-RG_DXE_msrvtt) |
| GL-RG + IT | MSR-VTT | DR | **46.9** | **31.2** | **65.7** | **60.6** | [GL-RG_DR_msrvtt](model/GL-RG_DR_msrvtt) |
| GL-RG | MSVD | XE | 52.3  | 33.8 | 70.4 | 58.7 | [GL-RG_XE_msvd](model/GL-RG_XE_msvd/model.pth) |
| GL-RG | MSVD | DXE | 57.7 | 38.6 | 74.9 | 95.9 | [GL-RG_DXE_msvd](model/GL-RG_DXE_msvd/model.pth) |
| GL-RG + IT | MSVD | DR | **60.5** | **38.9** | **76.4** | **101.0** | [GL-RG_DR_msvd](model/GL-RG_DR_msvd) |



## Test

Check out the trained model weights under the `model/` directory (following [Installation](docs/INSTALL.md)) and run:
```bash
./test.sh
```

**Note:** Please modify `MODEL_NAME`, `EXP_NAME` and `DATASET` in `test.sh` if experiment setting changes. For more details please refer to [TEST.md](docs/TEST.md).



## License

`GL-RG` is released under the MIT license.



## Acknowledgements
We are truly thankful of the following prior efforts in terms of knowledge contributions and open-source repos.
+ SA-LSTM: Describing Videos by Exploiting Temporal Structure (ICCV'15) [[paper]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Yao_Describing_Videos_by_ICCV_2015_paper.pdf) [[implement code]](https://github.com/hobincar/SA-LSTM)
+ RecNet: Reconstruction Network for Video Captioning (CVPR'18) [[paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Reconstruction_Network_for_CVPR_2018_paper.pdf) [[official code]](https://github.com/hobincar/RecNet) 
+ SAAT: Syntax-Aware Action Targeting for Video Captioning (CVPR'20) [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zheng_Syntax-Aware_Action_Targeting_for_Video_Captioning_CVPR_2020_paper.pdf) [[official code]](https://github.com/SydCaption/SAAT)
