# create on 2/17/2024
# author: Yiwen Wang
# test the models on the test set and save the results of the wavs
import argparse
import importlib
import multiprocessing
import os, glob
import logging
import json

import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm  # pylint: disable=unused-import
from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr,
    signal_distortion_ratio as sdr,
    scale_invariant_signal_distortion_ratio as si_sdr)

from tqdm import tqdm
from src.helpers import utils
from src.training.datasets.fsd50k_binaural import DatasetFsd50KPitchCond as Dataset
import scipy.io.wavfile as wavfile


def eval_wav(args):
    data_test = Dataset(args.test_csv)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False, num_workers=0)
    # Set up the device and workers.
    use_cuda = args.use_cuda and torch.cuda.is_available()
    if use_cuda:
        gpu_ids = args.gpu_ids if args.gpu_ids is not None\
                        else range(torch.cuda.device_count())
        device_ids = [_ for _ in gpu_ids]
        data_parallel = len(device_ids) > 1
        device = 'cuda:%d' % device_ids[0]
        torch.cuda.set_device(device_ids[0])
        logging.info("Using CUDA devices: %s" % str(device_ids))
    else:
        data_parallel = False
        device = torch.device('cpu')
        logging.info("Using device: CPU")
    # set up model 
    model = network.Net(**args.model_params)
    model_state = torch.load(args.best_ckpt)["model_state_dict"]
    new_state_dict = {}
    for key, value in model_state.items():
        new_key = key.replace("model.", "")
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)    
    # if necessary check gm filters
    gammatone_filters = model.gt_enc.filters()
    gammatone_filters_fc = model.gt_enc.fc
    gammatone_filters = gammatone_filters.detach().cpu().numpy()
    # sent to cuda
    if use_cuda and data_parallel:
        model = nn.DataParallel(model, device_ids=device_ids)
        logging.info("Using nn.DataParallel")
    model.to(device)

    with tqdm(total=len(test_loader), desc='test', ncols=100) as t:
        for batch_idx, (inp, tgt) in tqdm(enumerate(test_loader)):
            # Forward the network on the mixture.
            inp, tgt = test_loader.dataset.to(inp, tgt, device)
            net_output = model(inp)
            mix_save_path = os.path.join(args.save_folder, f"mix_{batch_idx}.wav")
            gt_save_path = os.path.join(args.save_folder, f"gt_{batch_idx}.wav")
            net_output_save_path = os.path.join(args.save_folder, f"net_output_{batch_idx}.wav")
            wavfile.write(mix_save_path, args.sr, inp['mixture'][0, 0, :].detach().cpu().numpy())
            wavfile.write(gt_save_path, args.sr, tgt[0, 0, :].detach().cpu().numpy())
            wavfile.write(net_output_save_path, args.sr, net_output['x'][0, 0, :].detach().cpu().numpy())
            t.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data Params
    parser.add_argument('exp_dir', type=str,
                        default='./experiments/fsd_mask_label_mult',
                        help="Path to save checkpoints and logs.")
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true',
                        help="Whether to use cuda")
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=None,
                        help="List of GPU ids used for training. "
                        "Eg., --gpu_ids 2 4. All GPUs are used by default.")
    parser.add_argument('--detect_anomaly', dest='detect_anomaly',
                        action='store_true',
                        help="Whether to use cuda")
    parser.add_argument('--best_ckpt', type=str, default=None,
                        help="Path to the best checkpoint")
    parser.add_argument('--wandb', dest='wandb', action='store_true',
                        help="Whether to sync tensorboard to wandb")
    parser.add_argument('--save_folder', type=str, default="L32Dim512Gammatone_save_pitch_condition_output",\
                        help="Path to save checkpoints and logs.")
    args = parser.parse_args()
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    # Load model and training params
    params = utils.Params(os.path.join(args.exp_dir, 'config.json'))
    for k, v in params.__dict__.items():
        vars(args)[k] = v

    exec("import %s as network" % args.model)
    logging.info("Imported the model from '%s'." % args.model)

    eval_wav(args)