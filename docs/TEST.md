## Test
Please modify `MODEL_NAME`, `EXP_NAME` and `DATASET` in `test.sh` according to the experiment settings.



### Example 1

If the GL-RG model is trained on MSR-VTT with DR reward, modify the relevant parts in `test.sh` into:

```ba
MODEL_NAME=GL-RG
EXP_NAME=DR         # Choices: [XE, DXE, DR]
DATASET=msrvtt      # Choices: [msrvtt, msvd]
```

The trained model weights `GL-RG_DR_msrvtt/model.pth` will be validated.

The output captions and scores will be written to: `model/GL-RG_DR_msrvtt/test_result.json`.



### Example 2

If the GL-RG model is trained on MSVD with XE loss, modify the relevant parts in `test.sh` into:

```ba
MODEL_NAME=GL-RG
EXP_NAME=XE         # Choices: [XE, DXE, DR]
DATASET=msvd        # Choices: [msrvtt, msvd]
```

The trained model weights `GL-RG_XE_msvd/model.pth` will be validated.

The output captions and scores will be written to: `model/GL-RG_XE_msvd/test_result.json`.



**(Optional)** Set `GPU_ID` to run with multi GPUs.

