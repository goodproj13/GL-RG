import argparse
import torch
import numpy as np
import os
import sys
import time
import math
import json

import logging
from datetime import datetime

from dataloader import DataLoader
from model import CaptionModel, CrossEntropyCriterion
from train import test

import utils
import opts

import requests

logger = logging.getLogger(__name__)

def progress_bar(some_iter):
    try:
        from tqdm import tqdm
        return tqdm(some_iter)
    except ModuleNotFoundError:
        return some_iter

def download_file_from_google_drive(file_id, destination):
    print("Trying to fetch {}".format(destination))

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in progress_bar(response.iter_content(CHUNK_SIZE)):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : file_id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : file_id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

ID_DICT = {'GL-RG_XE_msrvtt': '1xaAW-hUbOiXv5kdMxO-gCLAgl-wkre8q',
           'GL-RG_DXE_msrvtt': '1Jx1sCU2aQt0AA5-dRfRZallsCnbYB1Ud',
           'GL-RG_DR_msrvtt': '1x8Mh7HJuCmAWjwNExOR8MXqFCNyYttyJ',
           'GL-RG_XE_msvd': '1J4-I9bf2nB1_HlOLNq8aUpLfuSRvTlq3',
           'GL-RG_DXE_msvd': '1HixyH_LOT-3HtcsehQT_c9PAlPwFCTd-',
           'GL-RG_DR_msvd': '1cCisyMpp1mUS9NQPHSn4iCiVmqMfJeL5'}

if __name__ == '__main__':

    opt = opts.parse_opts()

    logging.basicConfig(level=getattr(logging, opt.loglevel.upper()),
                        format='%(asctime)s:%(levelname)s: %(message)s')

    logger.info('Input arguments: %s', json.dumps(vars(opt), sort_keys=True, indent=4))

    start = datetime.now()

    # test_opt = {'label_h5': opt.test_label_h5,
    #             'batch_size': opt.test_batch_size,
    #             'feat_h5': opt.test_feat_h5,
    #             'cocofmt_file': opt.test_cocofmt_file,
    #             'seq_per_img': opt.test_seq_per_img,
    #             'num_chunks': opt.num_chunks,
    #             'mode': 'test'
    #             }
    test_opt = {'label_h5': opt.test_label_h5,
                'batch_size': opt.test_batch_size,
                'feat_h5': opt.test_feat_h5,
                'cocofmt_file': opt.test_cocofmt_file,
                'seq_per_img': opt.test_seq_per_img,
                'num_chunks': opt.num_chunks,
                'use_resnet_feature': opt.use_resnet_feature,
                'use_c3d_feature': opt.use_c3d_feature,
                'use_audio_feature': opt.use_audio_feature,
                'use_sem_tag_feature': opt.use_sem_tag_feature,
                'use_long_range': opt.use_long_range,
                'use_short_range': opt.use_short_range,
                'use_local': opt.use_local,
                'mode': 'test'
                }

    test_loader = DataLoader(test_opt)

    if not os.path.exists(opt.model_file):
        logger.info('downloading model: %s', opt.model_file)
        model_name = opt.model_file.split('/')[-2]
        model_dir = os.path.dirname(opt.model_file)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        download_file_from_google_drive(ID_DICT[model_name], opt.model_file)

    logger.info('Loading model: %s', opt.model_file)
    checkpoint = torch.load(opt.model_file)
    checkpoint_opt = checkpoint['opt']

    opt.model_type = checkpoint_opt.model_type
    opt.vocab = checkpoint_opt.vocab
    opt.vocab_size = checkpoint_opt.vocab_size
    opt.seq_length = checkpoint_opt.seq_length
    opt.feat_dims = checkpoint_opt.feat_dims

    # assert opt.vocab_size == test_loader.get_vocab_size()
    assert opt.seq_length == test_loader.get_seq_length()
    assert opt.feat_dims == test_loader.get_feat_dims()

    logger.info('Building model...')
    model = CaptionModel(opt)
    logger.info('Loading state from the checkpoint...')
    model.load_state_dict(checkpoint['model'])

    xe_criterion = CrossEntropyCriterion()

    if torch.cuda.is_available():
        model.cuda()
        xe_criterion.cuda()

    logger.info('Start testing...')
    test(model, xe_criterion, test_loader, opt)
    logger.info('Time: %s', datetime.now() - start)
