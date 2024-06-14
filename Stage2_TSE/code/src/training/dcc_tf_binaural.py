import os
import math
from collections import OrderedDict
from typing import Optional
import logging
from copy import deepcopy

import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr,
    signal_distortion_ratio as sdr,
    scale_invariant_signal_distortion_ratio as si_sdr)

from src.training.dcc_tf import mod_pad, MaskNet
from src.helpers.eval_utils import itd_diff, ild_diff
from asteroid_filterbanks.enc_dec import Filterbank, Encoder, Decoder
import numpy as np 


class Param_GTFB(Filterbank):
    """ 
    Args:
        n_filters (int): Number of filters.
        kernel_size (int): Length of the filters.
        sample_rate (int, optional): The sample rate (used for initialization).
        stride (int, optional): Stride of the convolution. If None (default),
            set to ``kernel_size // 2``.
        
    p: order of gammatone (default = 4)
    fc: center frequency
    b: bandwidth
    phi: phase shift 
    """
    
    def __init__(self, n_filters=128, kernel_size=16, sample_rate=16000, stride=None, min_low_hz=50, min_band_hz=50, **kwargs):
        super().__init__(n_filters, kernel_size, stride=stride)
        self.sample_rate = sample_rate
        self.kernel_size = kernel_size
        self.n_feats_out = n_filters
        self.min_low_hz, self.min_band_hz = min_low_hz, min_band_hz
        
        _t_ = (torch.arange(1.0, self.kernel_size + 1).view(1, -1) / self.sample_rate)
                
        # filter order
        _p_ = torch.tensor(4.0, )  # order
        
        # intiliazation 
        low_hz = 50.0
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)
        # linear spaced in ERB scale
        erb_f = np.linspace(
            self.freq_hz_2_erb_scale(low_hz), self.freq_hz_2_erb_scale(high_hz), self.n_filters, dtype="float32"
        )  
        hz = self.erb_scale_2_freq_hz(erb_f) 
                        
        erb = 24.7 + 0.108 * hz  # equivalent rectangular bandwidth
        p_int_ = torch.tensor(4, dtype=torch.int32)  # order
        divisor = (np.pi * np.math.factorial(2 * p_int_ - 2) * np.power(2, float(-(2 * p_int_ - 2)))) / np.square(
                np.math.factorial(p_int_ - 1))
        _b_ = erb / divisor  # bandwidth parameter
        
        _phi_ = np.zeros(self.n_filters,dtype="float32")   
        
        self.register_buffer("_t_", _t_)
        
        # filters parameters         
        self.p = nn.Parameter(_p_.view(-1, 1), requires_grad=True)
        self.fc = nn.Parameter(torch.from_numpy(hz).view(-1, 1), requires_grad=True)       
        self.b = nn.Parameter(torch.from_numpy(_b_).view(-1, 1), requires_grad=True)
        self.phi = nn.Parameter(torch.from_numpy(_phi_).view(-1, 1), requires_grad=True)

    # @property
    def filters(self):
        
        eps = 1e-6
        
        phi_compensation = -self.fc * (self.p - 1) / (self.b+eps)    
    
        A = (4.0*np.pi*self.b)**((2.0*self.p+1.0)/2.0)/torch.sqrt(torch.exp(torch.lgamma(2.0*self.p+1.0)))*np.sqrt(2.0) # normalization
        gtone = self._t_**(self.p-1)*torch.exp(-2*np.pi*self.b*self._t_)*torch.cos(2*np.pi*self.fc*self._t_ + phi_compensation + self.phi)
        gtone = A * gtone
        normalization_value = 1.0 / torch.sqrt(torch.mean(torch.pow(gtone,2), dim=1))  # rms
        normalization_gtone = gtone * normalization_value[:, np.newaxis]        
        return normalization_gtone.view(self.n_filters, 1, self.kernel_size)

    
    @staticmethod
    def erb_scale_2_freq_hz(freq_erb):
        """ Convert frequency on ERB scale to frequency in Hertz """
        freq_hz = (np.exp(freq_erb / 9.265) - 1) * 24.7 * 9.265
        return freq_hz
    
    @staticmethod
    def freq_hz_2_erb_scale(freq_hz):
        """ Convert frequency in Hertz to frequency on ERB scale """
        freq_erb = 9.265 * np.log(1 + freq_hz / (24.7 * 9.265))
        return freq_erb 
    

class Net(nn.Module):
    def __init__(self, label_len, L=32,
                 model_dim=512, num_enc_layers=10,
                 dec_buf_len=100, num_dec_layers=2,
                 dec_chunk_size=72, out_buf_len=2,
                 use_pos_enc=True, conditioning="mult", lookahead=True,
                 pretrained_path=None):
        super(Net, self).__init__()
        self.L = L
        self.out_buf_len = out_buf_len
        self.model_dim = model_dim
        self.lookahead = lookahead
        self.freq_bin = 16000 // model_dim
        # Input conv to convert input audio to a latent representation
        kernel_size = 3 * L if lookahead else L
        self.gt_enc = Param_GTFB(n_filters=model_dim, kernel_size=kernel_size, sample_rate=16000, stride=L)
        self.gt_dec = Param_GTFB(n_filters=model_dim, kernel_size=kernel_size, sample_rate=16000, stride=L)
        self.enc = Encoder(self.gt_enc)
        self.dec = Decoder(self.gt_dec)
        self.in_conv = nn.Sequential(
            self.enc,
            nn.ReLU()
        )

        # Label embedding layer
        self.label_embedding = nn.Sequential(
            nn.Linear(label_len, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU())
        
        # convolve the conv1d feature and the pitch information
        self.LN = nn.GroupNorm(1, model_dim + model_dim // 2, eps=1e-8) #this is like layer normalization because the number of groups is equal to one
        self.BN = nn.Conv1d(model_dim + model_dim // 2, model_dim, 1)
        # Mask generator
        self.mask_gen = MaskNet(
            model_dim=model_dim, num_enc_layers=num_enc_layers,
            dec_buf_len=dec_buf_len,
            dec_chunk_size=dec_chunk_size, num_dec_layers=num_dec_layers,
            use_pos_enc=use_pos_enc, conditioning=conditioning)

        # Output conv layer
        self.out_conv = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=model_dim, out_channels=1,
                kernel_size=(out_buf_len + 1) * L,
                stride=L,
                padding=out_buf_len * L, bias=False),
            nn.Tanh())

        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path)['model_state_dict']
            # Load all the layers except label_embedding and freeze them
            for name, param in self.named_parameters():
                if 'label_embedding' not in name:
                    param.data = state_dict[name]
                    param.requires_grad = False

    def init_buffers(self, batch_size, device):
        enc_buf = self.mask_gen.encoder.init_ctx_buf(batch_size, device)
        dec_buf = self.mask_gen.decoder.init_ctx_buf(batch_size, device)
        out_buf = torch.zeros(batch_size, self.model_dim, self.out_buf_len,
                              device=device)
        return enc_buf, dec_buf, out_buf
    
    def fs2emb(self, x):
        # input x: seq of int value, size: [B, T]
        # output: sequence to the STFT contains f0
        emb_results = torch.zeros(x.shape[0], self.model_dim // 2, x.shape[-1], device=x.device)
        x_index = x // self.freq_bin 
        # TODO: update x with logdB index
        # convert x_index to LongIndex
        x_index = x_index.long()
        x_onehot = F.one_hot(x_index, num_classes=self.model_dim // 2).float()
        x_onehot = x_onehot.permute(0, 2, 1)
        return x_onehot

    def predict(self, x, label, pitch_info, enc_buf, dec_buf, out_buf):
        # Generate latent space representation of the input
        x = self.in_conv(x)
        pitch_info = F.interpolate(pitch_info.unsqueeze(1), size=x.shape[-1], mode='linear').squeeze(1)
        pitch_info = self.fs2emb(pitch_info)
        x = torch.cat((x, pitch_info), dim=1)
        x = self.BN(self.LN(x))
        # Generate label embedding
        l = self.label_embedding(label) # [B, label_len] --> [B, channels]
        l = l.unsqueeze(1).unsqueeze(-1) # [B, 1, channels, 1]
        # Generate mask corresponding to the label
        m, enc_buf, dec_buf = self.mask_gen(x, l, enc_buf, dec_buf)
        # Apply mask and decode
        x = x * m
        x = torch.cat((out_buf, x), dim=-1)
        out_buf = x[..., -self.out_buf_len:]
        x = self.out_conv(x)
        return x, enc_buf, dec_buf, out_buf

    def forward(self, inputs, init_enc_buf=None, init_dec_buf=None,
                init_out_buf=None, pad=True, writer=None, step=None, idx=None):
        """
        Extracts the audio corresponding to the `label` in the given
        `mixture`. Generates `chunk_size` samples per iteration.
        Args:
            mixed: [B, n_mics, T]
                input audio mixture
            label: [B, num_labels]
                one hot label
        Returns:
            out: [B, n_spk, T]
                extracted audio with sounds corresponding to the `label`
        """
        x, label, pitch_info = inputs['mixture'], inputs['label_vector'], inputs['pitch_info']
        if init_enc_buf is None or init_dec_buf is None or init_out_buf is None:
            assert init_enc_buf is None and \
                   init_dec_buf is None and \
                   init_out_buf is None, \
                "Both buffers have to initialized, or " \
                "both of them have to be None."
            enc_buf, dec_buf, out_buf = self.init_buffers(
                x.shape[0], x.device)
        else:
            enc_buf, dec_buf, out_buf = \
                init_enc_buf, init_dec_buf, init_out_buf

        mod = 0
        if pad:
            pad_size = (self.L, self.L) if self.lookahead else (0, 0)
            x, mod = mod_pad(x, chunk_size=self.L, pad=pad_size)
        # print("x shape: ", x.shape) # x shape:  torch.Size([16, 2, 48064])
        x, enc_buf, dec_buf, out_buf = self.predict(
            x, label, pitch_info, enc_buf, dec_buf, out_buf)

        # Remove mod padding, if present.
        if mod != 0:
            x = x[:, :, :-mod]
        
        out = {'x': x}

        if init_enc_buf is None:
            return out
        else:
            return out, enc_buf, dec_buf, out_buf

# Define optimizer, loss and metrics

def optimizer(model, data_parallel=False, **kwargs):
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.Adam(params, **kwargs)

def loss(_output, tgt):
    pred = _output['x']
    return -0.9 * snr(pred, tgt).mean() - 0.1 * si_snr(pred, tgt).mean()

def metrics(inputs, _output, gt):
    """ Function to compute metrics """
    mixed = inputs['mixture']
    output = _output['x']
    metrics = {}

    def metric_i(metric, src, pred, tgt):
        _vals = []
        for s, t, p in zip(src, tgt, pred):
            _vals.append(torch.mean((metric(p, t) - metric(s, t))).cpu().item())
        return _vals

    for m_fn in [snr, si_snr]:
        metrics[m_fn.__name__] = metric_i(m_fn,
                                          mixed[:, :gt.shape[1], :],
                                          output,
                                          gt)

    return metrics

def test_metrics(inputs, _output, gt):
    test_metrics = metrics(inputs, _output, gt)
    output = _output['x']
    delta_itds, delta_ilds, snrs = [], [], []
    for o, g in zip(output, gt):
        delta_itds.append(itd_diff(o.cpu(), g.cpu(), sr=44100))
        delta_ilds.append(ild_diff(o.cpu().numpy(), g.cpu().numpy()))
        snrs.append(torch.mean(si_snr(o, g)).cpu().item())
    test_metrics['delta_ITD'] = delta_itds
    test_metrics['delta_ILD'] = delta_ilds
    test_metrics['si_snr'] = snrs
    return test_metrics

def format_results(idx, inputs, output, gt, metrics, output_dir=None):
    results = metrics
    results['metadata'] = inputs['metadata']
    results = deepcopy(results)

    # Save audio
    if output_dir is not None:
        output = output['x']
        for i in range(output.shape[0]):
            out_dir = os.path.join(output_dir, f'{idx + i:03d}')
            os.makedirs(out_dir)
            torchaudio.save(
                os.path.join(out_dir, 'mixture.wav'), inputs['mixture'][i], 44100)
            torchaudio.save(
                os.path.join(out_dir, 'gt.wav'), gt[i], 44100)
            torchaudio.save(
                os.path.join(out_dir, 'output.wav'), output[i], 44100)

    return results

if __name__ == "__main__":
    torch.random.manual_seed(0)

    model = Net(41)
    model.eval()

    with torch.no_grad():
        x = torch.randn(1, 2, 417)
        emb = torch.randn(1, 41)

        y = model({'mixture': x, 'label_vector': emb})

        print(f'{y.shape=}')
        print(f"First channel data:\n{y[0, 0]}")
