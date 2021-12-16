#!/usr/bin/env bash

GPU_ID=0
MODEL_NAME=GL-RG
EXP_NAME=XE         # Choices: [XE, DXE, DR]
DATASET=msrvtt        # Choices: [msrvtt, msvd]

if [ "$DATASET" == "msvd" ]; then 
    SEQ_PER_IMG=17
else
    SEQ_PER_IMG=20
fi

func_GL-RG_testing()
{
        CUDA_VISIBLE_DEVICES=${GPU_ID} python test.py    --model_file model/${MODEL_NAME}_${EXP_NAME}_${DATASET}/model.pth \
                                                        --test_label_h5 data/metadata/${DATASET}_test_sequencelabel.h5 \
                                                        --test_cocofmt_file data/metadata/${DATASET}_test_cocofmt.json \
                                                        --test_feat_h5  "" "" "" data/feature/${DATASET}_test_sem_tag_res.h5 \
                                                        --use_resnet_feature 0 \
                                                        --use_c3d_feature 0 \
                                                        --use_audio_feature 0 \
                                                        --use_sem_tag_feature 1 \
                                                        --use_long_range 1 \
                                                        --use_short_range 1 \
                                                        --use_local 1 \
                                                        --beam_size 5 \
                                                        --language_eval 1 \
                                                        --test_seq_per_img ${SEQ_PER_IMG} \
                                                        --test_batch_size 64 \
                                                        --loglevel INFO \
                                                        --result_file model/${MODEL_NAME}_${EXP_NAME}_${DATASET}/test_result.json
}

func_GL-RG_testing
