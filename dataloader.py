from __future__ import print_function

import torch
import json
import h5py
import os
import numpy as np
import random
import time
import cPickle

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DataLoader():
    """Class to load video features and captions"""

    def __init__(self, opt):

        self.iterator = 0
        self.epoch = 0

        self.batch_size = opt.get('batch_size', 128)
        self.seq_per_img = opt.get('seq_per_img', 1)
        self.word_embedding_size = opt.get('word_embedding_size', 512)
        self.num_chunks = opt.get('num_chunks', 1)
        self.mode = opt.get('mode', 'train')
        self.cocofmt_file = opt.get('cocofmt_file', None)
        self.bcmrscores_pkl = opt.get('bcmrscores_pkl', None)

        self.use_c3d_feature = opt.get('use_c3d_feature', 0)
        self.use_audio_feature = opt.get('use_audio_feature', 0)
        self.use_sem_tag_feature = opt.get('use_sem_tag_feature', 0)
        self.use_resnet_feature = opt.get('use_resnet_feature', 0)

        # open the hdf5 info file
        logger.info('DataLoader loading h5 file: %s', opt['label_h5'])
        self.label_h5 = h5py.File(opt['label_h5'], 'r')

        self.vocab = [i for i in self.label_h5['vocab']]
        self.videos = [i for i in self.label_h5['videos']]

        self.ix_to_word = {i: w for i, w in enumerate(self.vocab)}
        self.num_videos = len(self.videos)
        self.index = range(self.num_videos)

        # load the json file which contains additional information about the
        # dataset
        self.feat_h5_files = opt['feat_h5']
        logger.info('DataLoader loading h5 files: %s', self.feat_h5_files)
        if self.use_resnet_feature == 1:
            self.feat_h5_res = h5py.File(self.feat_h5_files[0], 'r')
        if self.use_c3d_feature == 1:
            self.feat_h5_c3d = h5py.File(self.feat_h5_files[1], 'r')
        if self.use_audio_feature == 1:
            self.feat_h5_aud = h5py.File(self.feat_h5_files[2], 'r')
        #self.feat_h5_cls = h5py.File(self.feat_h5_files[3], 'r')
        if self.use_sem_tag_feature == 1:
            self.feat_h5_sem_tag = h5py.File(self.feat_h5_files[3], 'r')

        self.feat_dims = []
        if self.use_resnet_feature == 1:
            self.feat_dim_res = self.feat_h5_res[self.videos[0]].shape[0] if 'mp1' in self.feat_h5_files[0] else self.feat_h5_res['feats'][0].shape[1]
        if self.use_c3d_feature == 1:
            self.feat_dim_c3d = self.feat_h5_c3d[self.videos[0]].shape[0] if 'mp1' in self.feat_h5_files[1] else self.feat_h5_c3d['feats'][0].shape[0]
        if self.use_audio_feature == 1:
            self.feat_dim_aud = self.feat_h5_aud[self.videos[0]].shape[0] if 'mp1' in self.feat_h5_files[2] else self.feat_h5_aud['feats'][0].shape[0]
        if self.use_sem_tag_feature == 1:
            self.feat_dim_sem_tag = self.feat_h5_sem_tag[self.videos[0]].shape[0] if 'mp1' in self.feat_h5_files[3] else self.feat_h5_sem_tag['feats'][0].shape[0]

        #self.feat_dim_cls = self.feat_h5_cls[self.videos[0]].shape[0]
        if self.use_resnet_feature == 1:
            self.feat_dims.append(self.feat_dim_res)
        if self.use_c3d_feature == 1:
            self.feat_dims.append(self.feat_dim_c3d)
        if self.use_audio_feature == 1:
            self.feat_dims.append(self.feat_dim_aud)
        #self.feat_dims.append(self.feat_dim_cls)
        if self.use_sem_tag_feature == 1:
            self.feat_dims.append(self.feat_dim_sem_tag)

        self.num_feats = len(self.feat_h5_files)
        if self.use_resnet_feature == 0:
            self.num_feats = len(self.feat_h5_files) - 1
        if self.use_c3d_feature == 0:
            self.num_feats = len(self.feat_h5_files) - 1
        if self.use_audio_feature == 0:
            self.num_feats = len(self.feat_h5_files) - 1
        if self.use_sem_tag_feature == 0:
            self.num_feats = self.num_feats - 1

        # load in the sequence data
        if 'labels' in self.label_h5.keys():
            self.seq_length = self.label_h5['labels'].shape[1]
            logger.info('max sequence length in data is: %d', self.seq_length)

            # load the pointers in full to RAM (should be small enough)
            self.label_start_ix = self.label_h5['label_start_ix']
            self.label_end_ix = self.label_h5['label_end_ix']
            assert (self.label_start_ix.shape[0] == self.label_end_ix.shape[0])
            self.has_label = True
        else:
            self.has_label = False

        if self.bcmrscores_pkl is not None:
            eval_metric = opt.get('eval_metric', 'CIDEr')
            logger.info('Loading: %s, with metric: %s', self.bcmrscores_pkl, eval_metric)
            self.bcmrscores = cPickle.load(open(self.bcmrscores_pkl))
            if eval_metric == 'CIDEr' and eval_metric not in self.bcmrscores:
                eval_metric = 'cider'
            self.bcmrscores = self.bcmrscores[eval_metric]

        if self.mode == 'train':
            self.shuffle_videos()

    def __del__(self):
        if self.use_resnet_feature == 1:
            self.feat_h5_res.close()
        if self.use_c3d_feature == 1:
            self.feat_h5_c3d.close()
        if self.use_audio_feature == 1:
            self.feat_h5_aud.close()
        #self.feat_h5_cls.close()
        if self.use_sem_tag_feature == 1:
            self.feat_h5_sem_tag.close()

        self.label_h5.close()

    def update_index(self, video_id, filename):
        if self.cocofmt_file.find("msrvtt") != -1:
            if 'train' in filename:
                releative_id = video_id
            if 'val' in filename:
                releative_id = video_id - 6513
            if 'test' in filename:
                releative_id = video_id - 6513 - 497
        elif self.cocofmt_file.find("msvd") != -1:
            if 'train' in filename:
                releative_id = video_id
            if 'val' in filename:
                releative_id = video_id - 1200
            if 'test' in filename:
                releative_id = video_id - 1200 - 100
        else:
            raise "Please use MSVD or MSR-VTT"

        return releative_id

    def get_batch(self):
        video_batchs = []
        if self.use_resnet_feature == 1:
            video_batch_res = torch.FloatTensor(self.batch_size, self.num_chunks, 20, self.feat_dim_res).zero_()
        if self.use_c3d_feature == 1:
            video_batch_c3d = torch.FloatTensor(self.batch_size, self.num_chunks, self.feat_dim_c3d).zero_()
        if self.use_audio_feature == 1:
            video_batch_aud = torch.FloatTensor(self.batch_size, self.num_chunks, self.feat_dim_aud).zero_()
        #video_batch_cls = torch.FloatTensor(self.batch_size, self.num_chunks, self.feat_dim_cls).zero_()
        if self.use_sem_tag_feature == 1:
            video_batch_sem_tag = torch.FloatTensor(self.batch_size, self.num_chunks, self.feat_dim_sem_tag).zero_()

        if self.use_resnet_feature == 1:
            video_batchs.append(video_batch_res)
        if self.use_c3d_feature == 1:
            video_batchs.append(video_batch_c3d)
        if self.use_audio_feature == 1:
            video_batchs.append(video_batch_aud)
        #video_batchs.append(video_batch_cls)
        if self.use_sem_tag_feature == 1:
            video_batchs.append(video_batch_sem_tag)

        if self.has_label:
            label_batch = torch.LongTensor(self.batch_size * self.seq_per_img, self.seq_length).zero_()
            mask_batch = torch.FloatTensor(self.batch_size * self.seq_per_img, self.seq_length).zero_()

        videoids_batch = []
        gts = []
        bcmrscores = np.zeros((self.batch_size, self.seq_per_img)) if self.bcmrscores_pkl is not None else None

        for ii in range(self.batch_size):
            idx = self.index[self.iterator]
            video_id = int(self.videos[idx])
            videoids_batch.append(video_id)

            feat_idx = 0

            if self.use_resnet_feature == 1:
                if 'mp1' in self.feat_h5_files[0]:
                    video_batchs[feat_idx][ii] = torch.from_numpy(np.array(self.feat_h5_res[str(video_id)]))
                else:
                    video_batchs[feat_idx][ii] = torch.from_numpy(
                        np.array(self.feat_h5_res['feats'][self.update_index(video_id, self.feat_h5_files[0])]))
                feat_idx += 1

            if self.use_c3d_feature == 1:
                if 'mp1' in self.feat_h5_files[1]:
                    video_batchs[feat_idx][ii] = torch.from_numpy(np.array(self.feat_h5_c3d[str(video_id)]))
                else:
                    video_batchs[feat_idx][ii] = torch.from_numpy(np.array(self.feat_h5_c3d['feats'][self.update_index(video_id, self.feat_h5_files[1])]))
                feat_idx += 1

            if self.use_audio_feature == 1:
                if 'mp1' in self.feat_h5_files[2]:
                    video_batchs[feat_idx][ii] = torch.from_numpy(np.array(self.feat_h5_aud[str(video_id)]))
                else:
                    video_batchs[feat_idx][ii] = torch.from_numpy(np.array(self.feat_h5_aud['feats'][self.update_index(video_id, self.feat_h5_files[2])]))
                feat_idx += 1

            if self.use_sem_tag_feature == 1:
                if 'mp1' in self.feat_h5_files[3]:
                    video_batchs[feat_idx][ii] = torch.from_numpy(np.array(self.feat_h5_sem_tag[str(video_id)]))
                else:
                    video_batchs[feat_idx][ii] = torch.from_numpy(np.array(self.feat_h5_sem_tag['feats'][self.update_index(video_id, self.feat_h5_files[3])]))
                feat_idx += 1


            '''
            if 'mp1' in self.feat_h5_files[3]:
                video_batchs[3][ii] = torch.from_numpy(np.array(self.feat_h5_cls[str(video_id)]))
            else:
                video_batchs[3][ii] = torch.from_numpy(np.array(self.feat_h5_cls[str(video_id)]))
            '''

            if self.has_label:
                # fetch the sequence labels
                ix1 = self.label_start_ix[idx]
                ix2 = self.label_end_ix[idx]
                ncap = ix2 - ix1  # number of captions available for this image
                assert ncap > 0, 'No captions!!'

                seq = torch.LongTensor(self.seq_per_img, self.seq_length).zero_()
                seq_all = torch.from_numpy(np.array(self.label_h5['labels'][ix1:ix2]))

                if ncap <= self.seq_per_img:
                    seq[:ncap] = seq_all[:ncap]
                    for q in range(ncap, self.seq_per_img):
                        ixl = np.random.randint(ncap)
                        seq[q] = seq_all[ixl]
                else:
                    randpos = torch.randperm(ncap)
                    for q in range(self.seq_per_img):
                        ixl = randpos[q]
                        seq[q] = seq_all[ixl]

                il = ii * self.seq_per_img
                label_batch[il:il + self.seq_per_img] = seq

                # Used for reward evaluation
                gts.append(self.label_h5['labels'][self.label_start_ix[idx]: self.label_end_ix[idx]])

                # pre-computed cider scores, 
                # assuming now that videos order are same (which is the sorted videos order)
                if self.bcmrscores_pkl is not None:
                    bcmrscores[ii] = self.bcmrscores[idx]

            self.iterator += 1
            if self.iterator >= self.num_videos:
                logger.info('===> Finished loading epoch %d', self.epoch)
                self.iterator = 0
                self.epoch += 1
                if self.mode == 'train':
                    self.shuffle_videos()

        data = {}
        data['feats'] = video_batchs
        data['ids'] = videoids_batch

        if self.has_label:
            # + 1 here to count the <eos> token, because the <eos> token is set to 0
            nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 1, label_batch)))
            for ix, row in enumerate(mask_batch):
                row[:nonzeros[ix]] = 1

            data['labels'] = label_batch
            data['masks'] = mask_batch
            data['gts'] = gts
            data['bcmrscores'] = bcmrscores

        return data

    def reset(self):
        self.iterator = 0

    def get_current_index(self):
        return self.iterator

    def set_current_index(self, index):
        self.iterator = index

    def get_vocab(self):
        return self.ix_to_word

    def get_vocab_size(self):
        return len(self.vocab)

    def get_feat_dims(self):
        return self.feat_dims

    def get_feat_size(self):
        return sum(self.feat_dims)

    def get_num_feats(self):
        return self.num_feats

    def get_seq_length(self):
        return self.seq_length

    def get_seq_per_img(self):
        return self.seq_per_img

    def get_num_videos(self):
        return self.num_videos

    def get_batch_size(self):
        return self.batch_size

    def get_current_epoch(self):
        return self.epoch

    def set_current_epoch(self, epoch):
        self.epoch = epoch

    def shuffle_videos(self):
        np.random.shuffle(self.index)

    def get_cocofmt_file(self):
        return self.cocofmt_file
