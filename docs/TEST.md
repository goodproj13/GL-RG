## Test
Please modify `MODEL_NAME`, `EXP_NAME` and `DATASET` in `test.sh` according to the experiment settings.

For example, if the GL-RG model is trained on MSR-VTT with XE loss, modify the relevant parts in `test.sh` into:

```ba
MODEL_NAME=GL-RG
EXP_NAME=XE         # Choices: [XE, DXE, DR]
DATASET=msrvtt      # Choices: [msrvtt, msvd]
```

The trained model weights `GL-RG_XE_msrvtt/model.pth` will be validated.

