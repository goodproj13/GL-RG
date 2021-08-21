import sys
import os
import json

import numpy as np
from collections import OrderedDict

sys.path.append("cider")
from pyciderevalcap.ciderD.ciderD import CiderD

sys.path.append('coco-caption')
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor

import cPickle


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every [lr_update] epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if isinstance(score, list):
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


def load_gt_refs(cocofmt_file):
    d = json.load(open(cocofmt_file))
    out = {}
    for i in d['annotations']:
        out.setdefault(i['image_id'], []).append(i['caption'])
    return out


def compute_score(gt_refs, predictions, scorer):
    # use with standard package https://github.com/tylin/coco-caption
    # hypo = {p['image_id']: [p['caption']] for p in predictions}

    # use with Cider provided by https://github.com/ruotianluo/cider
    hypo = [{'image_id': p['image_id'], 'caption': [p['caption']]}
            for p in predictions]

    # standard package requires ref and hypo have same keys, i.e., ref.keys()
    # == hypo.keys()
    ref = {p['image_id']: gt_refs[p['image_id']] for p in predictions}

    score, scores = scorer.compute_score(ref, hypo)

    return score, scores


# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j]
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                ix = int(ix)
                txt = txt + ix_to_word[ix]
            else:
                break
        out.append(txt)
    return out


# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def compute_avglogp(seq, logseq, eos_token=0):
    seq = seq.cpu().numpy()
    logseq = logseq.cpu().numpy()

    N, D = seq.shape
    out_avglogp = []
    for i in range(N):
        avglogp = []
        for j in range(D):
            ix = seq[i, j]
            avglogp.append(logseq[i, j])
            if ix == eos_token:
                break
        avg = 0 if len(avglogp) == 0 else sum(avglogp) / float(len(avglogp))
        out_avglogp.append(avg)
    return out_avglogp


def language_eval(gold_file, pred_file):
    # save the current stdout
    temp = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    coco = COCO(gold_file)
    cocoRes = coco.loadRes(pred_file)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = round(score, 5)

    # restore the previous stdout
    sys.stdout = temp
    return out


def array_to_str(arr, use_eos=0):
    out = ''
    for i in range(len(arr)):
        if use_eos == 0 and arr[i] == 0:
            break

        # skip the <bos> token    
        if arr[i] == 1:
            continue

        out += str(arr[i]) + ' '

        # return if encouters the <eos> token
        # this will also guarantees that the first <eos> will be rewarded
        if arr[i] == 0:
            break

    return out.strip()
