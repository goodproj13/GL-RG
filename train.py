import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
import numpy as np
import os
import sys
import time
import math
import json
import uuid
import logging
from datetime import datetime
from six.moves import cPickle

from dataloader import DataLoader
from model import CaptionModel, CrossEntropyCriterion, RewardCriterion

from data.setup import setup
setup()

import utils
import opts

import sys


sys.path.append("cider")
from pyciderevalcap.cider.cider import Cider
from pyciderevalcap.ciderD.ciderD import CiderD

sys.path.append('coco-caption')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

logger = logging.getLogger(__name__)


def language_eval(predictions, cocofmt_file, opt):
    logger.info('>>> Language evaluating ...')
    tmp_checkpoint_json = os.path.join(
        opt.model_file + str(uuid.uuid4()) + '.json')
    json.dump(predictions, open(tmp_checkpoint_json, 'w'))
    lang_stats = utils.language_eval(cocofmt_file, tmp_checkpoint_json)
    os.remove(tmp_checkpoint_json)
    return lang_stats


def validate(model, criterion, loader, opt):
    model.eval()
    loader.reset()

    num_videos = loader.get_num_videos()
    batch_size = loader.get_batch_size()
    num_iters = int(math.ceil(num_videos * 1.0 / batch_size))
    last_batch_size = num_videos % batch_size
    seq_per_img = loader.get_seq_per_img()
    model.set_seq_per_img(seq_per_img)

    loss_sum = 0
    logger.info('#num_iters: %d, batch_size: %d, seg_per_image: %d', num_iters, batch_size, seq_per_img)
    predictions = []
    gt_avglogps = []
    test_avglogps = []
    for ii in range(num_iters):
        data = loader.get_batch()
        feats = [Variable(feat, volatile=True) for feat in data['feats']]
        if loader.has_label:
            labels = Variable(data['labels'], volatile=True)
            masks = Variable(data['masks'], volatile=True)

        if ii == (num_iters - 1) and last_batch_size > 0:
            feats = [f[:last_batch_size] for f in feats]
            if loader.has_label:
                labels = labels[:last_batch_size * seq_per_img]  # labels shape is DxN
                masks = masks[:last_batch_size * seq_per_img]

        if torch.cuda.is_available():
            feats = [feat.cuda() for feat in feats]
            if loader.has_label:
                labels = labels.cuda()
                masks = masks.cuda()

        if loader.has_label:
            t_start = time.time()
            pred, gt_seq, gt_logseq = model(feats, labels)
            logger.info("Inference time: %f, batch_size: %d" % ((time.time() - t_start) / batch_size, batch_size))
            if opt.output_logp == 1:
                gt_avglogp = utils.compute_avglogp(gt_seq, gt_logseq.data)
                gt_avglogps.extend(gt_avglogp)

            loss = criterion(pred, labels[:, 1:], masks[:, 1:])
            if float(torch.__version__[:3]) > 0.5:
                loss_sum += loss.item()
            else:
                loss_sum += loss.data[0]
            

        
        seq, logseq = model.sample(feats, {'beam_size': opt.beam_size})

        
        sents = utils.decode_sequence(opt.vocab, seq)
        if opt.output_logp == 1:
            test_avglogp = utils.compute_avglogp(seq, logseq)
            test_avglogps.extend(test_avglogp)

        for jj, sent in enumerate(sents):
            if opt.output_logp == 1:
                entry = {'image_id': data['ids'][jj], 'caption': sent, 'avglogp': test_avglogp[jj]}
            else:
                entry = {'image_id': data['ids'][jj], 'caption': sent}
            predictions.append(entry)
            logger.debug('[%d] video %s: %s' % (jj, entry['image_id'], entry['caption']))

    loss = round(loss_sum / num_iters, 3)
    results = {}
    lang_stats = {}

    if opt.language_eval == 1 and loader.has_label:
        logger.info('>>> Language evaluating ...')
        tmp_checkpoint_json = os.path.join(opt.model_file + str(uuid.uuid4()) + '.json')
        json.dump(predictions, open(tmp_checkpoint_json, 'w'))
        lang_stats = utils.language_eval(loader.cocofmt_file, tmp_checkpoint_json)
        os.remove(tmp_checkpoint_json)

    results['predictions'] = predictions
    results['scores'] = {'Loss': -loss}
    results['scores'].update(lang_stats)

    if opt.output_logp == 1:
        avglogp = sum(test_avglogps) / float(len(test_avglogps))
        results['scores'].update({'avglogp': avglogp})

        gt_avglogps = np.array(gt_avglogps).reshape(-1, seq_per_img)
        assert num_videos == gt_avglogps.shape[0]

        gt_avglogps_file = opt.model_file.replace('.pth', '_gt_avglogps.pkl', 1)
        cPickle.dump(gt_avglogps, open(gt_avglogps_file, 'w'), protocol=cPickle.HIGHEST_PROTOCOL)

        logger.info('Wrote GT logp to: %s', gt_avglogps_file)

    return results


def test(model, criterion, loader, opt):
    results = validate(model, criterion, loader, opt)
    logger.info('Test output: %s', json.dumps(results['scores'], indent=4))

    json.dump(results, open(opt.result_file, 'w'))
    logger.info('Wrote output caption to: %s ', opt.result_file)

