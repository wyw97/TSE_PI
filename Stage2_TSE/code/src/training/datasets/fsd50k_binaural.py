import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import hub
from torch.utils.data import Dataset, DataLoader
import random as random
import os
import shutil
import zipfile
import json
import soundfile as sf 
import torchaudio 
import torchaudio.transforms as AT
import matplotlib.pyplot as plt 


class DatasetFsd50KPitchCond(Dataset):
    """Dataset class for LibriMix source separation tasks.

    Args:
        csv_dir (str): The path to the metadata file.
        task (str): One of ``'enh_single'``, ``'enh_both'``, ``'sep_clean'`` or
            ``'sep_noisy'`` :

            * ``'enh_single'`` for single speaker speech enhancement.
            * ``'enh_both'`` for multi speaker speech enhancement.
            * ``'sep_clean'`` for two-speaker clean source separation.
            * ``'sep_noisy'`` for two-speaker noisy source separation.

        sample_rate (int) : The sample rate of the sources and mixtures.
        n_src (int) : The number of sources in the mixture.
        segment (int, optional) : The desired sources and mixtures length in s.

    References
        [1] "LibriMix: An Open-Source Dataset for Generalizable Speech Separation",
        Cosentino et al. 2020.
    """

    dataset_name = "DatasetFsd50KPitchCond"

    def __init__(
          self, csv_dir, sample_rate=16000, n_src=2, segment=3, return_id=False, \
          cate2id="fsd50_cate2id.json",
    ):
        self.csv_dir = csv_dir
        self.return_id = return_id
        self.segment = segment
        self.sr = sample_rate
        self.seg_len = int(self.segment * self.sr)
        # Open csv file
        self.path_files = []
        with open(self.csv_dir, "r") as wf:
            for line in wf.readlines():
                self.path_files.append(line.strip())
        # each path file contains two paths: mixture_path and source_path
        self.n_src = n_src
        self.cate2id = dict({})
        with open(cate2id, "r") as f:
            for line in f.readlines():
                line = line.strip()
                self.cate2id[line.split(":")[0]] = int(line.split(":")[1])
        self.cate_num = len(self.cate2id.keys())

    def __len__(self):
        return len(self.path_files)


    def __getitem__(self, idx):
        # Get the row in dataframe
        path_info_line = self.path_files[idx] 
        s1_path, s2_path = path_info_line.strip().split(";")[0], path_info_line.strip().split(";")[1]
        keyID = int(path_info_line.split(";")[-1])
        pitch_path = path_info_line.split(";")[-2]
        
        sources_list = []
        s1_wav, _ = sf.read(s1_path, dtype='float32')
        s2_wav, _ = sf.read(s2_path, dtype='float32')
        # label one hot
        label = np.zeros((self.cate_num, ), dtype=np.float32)
        label[keyID] = 1.0
        # label = torch.from_numpy(label)
        # load pitch (f0) information
        pitch_info = np.load(pitch_path)
        sources_list.append(s1_wav[:, 0])
        sources = np.vstack(sources_list)
        mix_wav = (s1_wav[:, 0] + s2_wav[:, 0]) / 2
        inputs = dict({
            'mixture': torch.from_numpy(mix_wav.reshape(-1, self.seg_len)),
            'label_vector': torch.from_numpy(label),
            'pitch_info': pitch_info
        })
        return inputs, torch.from_numpy(sources.reshape(-1, self.seg_len))
    
    def _dataset_name(self):
        return f"DatasetPitchCond{self.n_src}"

    def to(self, inputs, gt, device):
        inputs['mixture'] = inputs['mixture'].to(device)
        inputs['label_vector'] = inputs['label_vector'].to(device)
        inputs['pitch_info'] = inputs['pitch_info'].to(device)
        gt = gt.to(device)
        return inputs, gt

    def output_to(self, output, device):
        for k, v in output.items():
            output[k] = v.to(device)
        return output

    def output_detach(self, output):
        for k, v in output.items():
            output[k] = v.detach()
        return output

    def collate_fn(self, batch):
        inputs, gt = zip(*batch)
        inputs = {
            'mixture': torch.stack([i['mixture'] for i in inputs]),
            'label_vector': torch.stack([i['label_vector'] for i in inputs]),
            'pitch_info': torch.stack([i['pitch_info'] for i in inputs]),
            # 'shift': torch.stack([i['shift'] for i in inputs]),
            'metadata': [i['metadata'] for i in inputs]
        }
        gt = torch.stack(gt)
        return inputs, gt

    def tensorboard_add_metrics(self, writer, tag, metrics, step):
        """
        Add metrics to tensorboard.
        """
        vals = np.asarray(metrics['_scale_invariant_signal_noise_ratio'])
        writer.add_histogram('%s/%s' % (tag, 'SI-SNRi'), np.array(vals), step)
        return