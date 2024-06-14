import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
from torchaudio import transforms
from collections import OrderedDict
import torch.nn.utils.weight_norm as wn
import math
#from  memory_profiler import profile
from typing import TypedDict
import torch.nn.functional as F

PI = torch.Tensor([math.pi])
    

class InputSpec(nn.Module):
    def __init__(self, n_fft=512, hop_length=256, win_length=512) -> None:
        super().__init__()
        self.spec= transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                      window_fn=torch.hann_window, power=None) 
        
    def forward(self, sig_time):
        stft = self.spec(sig_time)
        stft[:, 0, :] = 0 #remove DC
        return stft 
    
    
class cLN(nn.Module):
    def __init__(self, dimension, eps=1e-8, trainable=True):
        """not being used in our implementation. used for causal case
        """
        super(cLN, self).__init__()
        self.eps = eps
        if trainable:

            self.gain = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step

        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)

        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T

        entry_cnt = np.arange(channel, channel * (time_step + 1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T

        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)

        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class DepthConv1d(nn.Module):
    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=False, causal=False, bool_drop=True,
                 drop_value=0.1, weight_norm=False):
        super(DepthConv1d, self).__init__()
        self.causal = causal
        self.skip = skip
        self.bool_drop = bool_drop
        if not weight_norm:
            self.conv1d = nn.Conv1d(input_channel, input_channel, 1)

            if self.causal:
                self.padding = (kernel - 1) * dilation
            else:
                self.padding = padding
            groups = int(hidden_channel/4) #int(hidden_channel/2) #for BN=256 #TODO
            self.dconv1d = nn.Conv1d(input_channel, hidden_channel, kernel, dilation=dilation,
                                    groups=groups,
                                    padding=self.padding) #Depth-wise convolution
            self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
            self.nonlinearity1 = nn.PReLU()
            self.nonlinearity2 = nn.PReLU()
            self.drop1 = nn.Dropout2d(drop_value)
            self.drop2 = nn.Dropout2d(drop_value)
            if self.causal:
                self.reg1 = cLN(hidden_channel, eps=1e-08)
                self.reg2 = cLN(hidden_channel, eps=1e-08)
            else:
                self.reg1 = nn.GroupNorm(1, input_channel, eps=1e-08)
                self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)

            if self.skip:
                self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

        else:
            self.conv1d = wn(nn.Conv1d(input_channel, input_channel, 1))

            if self.causal:
                self.padding = (kernel - 1) * dilation
            else:
                self.padding = padding
            groups = input_channel
            self.dconv1d = wn(nn.Conv1d(input_channel, hidden_channel, kernel, dilation=dilation,
                                    groups=groups,
                                    padding=self.padding)) #Depth-wise convolution
            self.res_out = wn(nn.Conv1d(hidden_channel, input_channel, 1))
            self.nonlinearity1 = nn.PReLU()
            self.nonlinearity2 = nn.PReLU()
            self.drop1 = nn.Dropout2d(drop_value)
            self.drop2 = nn.Dropout2d(drop_value)
            if self.causal:
                self.reg1 = cLN(input_channel, eps=1e-08)
                self.reg2 = cLN(hidden_channel, eps=1e-08)
            else:
                self.reg1 = nn.GroupNorm(1, input_channel, eps=1e-08)
                self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)

            if self.skip:
                self.skip_out = wn(nn.Conv1d(hidden_channel, input_channel, 1))

    #@profile
    def forward(self, input):
        if self.bool_drop:
            output = self.reg1(self.drop1(self.nonlinearity1(self.conv1d(input))))#with dropout
            if self.causal:
                output = self.reg2(self.drop2(self.nonlinearity2(self.dconv1d(output)[:, :, :-self.padding])))
            else:
                output = self.reg2(self.drop2(self.nonlinearity2(self.dconv1d(output))))
        else:
            output = self.reg1(self.nonlinearity1(self.conv1d(input)))#without droput
            if self.causal:
                output = self.reg2(self.nonlinearity2(self.dconv1d(output)[:, :, :-self.padding]))
            else:
                output = self.reg2(self.nonlinearity2(self.dconv1d(output))) #without dropout 
        
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual #In our implementation we don't use skip connection
            

class TF_Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1d_t_1 = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv1d_t_2 = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=2, dilation=2)
        self.sigmoid_t = nn.Sigmoid()
        self.prelu_t = nn.PReLU()
        self.adapt_avrg_pooling_t = nn.AdaptiveAvgPool2d((1, None))
        
        self.conv1d_f_1 = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv1d_f_2 = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=2, dilation=2)
        self.sigmoid_f = nn.Sigmoid()
        self.prelu_f = nn.PReLU()
        self.adapt_avrg_pooling_f = nn.AdaptiveAvgPool2d((None, 1))
        
    def forward(self, input):
        output_t = self.adapt_avrg_pooling_t(input) #[B, 1, T]
        output_t = self.sigmoid_t(self.prelu_t(self.conv1d_t_2(self.conv1d_t_1(output_t))))
        
        output_f = self.adapt_avrg_pooling_f(input) #[B, F, 1]
        output_f = torch.transpose(output_f, 1, 2) #[B, 1, F]
        output_f = self.sigmoid_f(self.prelu_f(self.conv1d_f_2(self.conv1d_f_1(output_f))))
        output_f = torch.transpose(output_f, 1, 2) #[B, F, 1]
        
        attention_w = output_f @ output_t #[B, F, T]
        output = input * attention_w 
        return output


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FiLMLayer, self).__init__()
        self.fc_gamma = nn.Linear(input_dim, output_dim)
        self.fc_beta = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        # add ReLU for each layer
        gamma = self.fc_gamma(x)
        beta = self.fc_beta(x)
        gamma = F.relu(gamma)
        beta = F.relu(beta)
        return gamma.unsqueeze(-1), beta.unsqueeze(-1)


class TCN_Film(nn.Module): #this is the audio blocks add the film condition
    def __init__(self, input_dim, output_dim, BN_dim, H_dim,
                layer, R, kernel=3, skip=False,
                causal=False, dilated=True, bool_drop=True, drop_value=0.1, weight_norm=False,
                tf_attention=False, apply_recursive_ln=False, apply_residual_ln=False, film_input=27):
        super(TCN_Film, self).__init__()
        self.layer = layer
        self.tf_attention = tf_attention
        self.apply_recursive_ln = apply_recursive_ln
        self.apply_residual_ln = apply_residual_ln
        self.skip = skip
        self.dnn_layers = nn.ModuleList([])
        # normalization
        if not weight_norm:
            if not causal:
                self.LN = nn.GroupNorm(1, input_dim, eps=1e-8) #this is like layer normalization because the number of groups is equal to one
            else:
                self.LN = cLN(input_dim, eps=1e-8)

            self.BN = nn.Conv1d(input_dim, BN_dim, 1)
            
            # TCN for feature extraction
            self.receptive_field = 0
            self.dilated = dilated

            self.TCN = nn.ModuleList([])
            if self.tf_attention:
                self.time_freq_attnetion = nn.ModuleList([])
            for r in range(R):
                for i in range(layer):
                    if self.dilated:
                        if i == 0:
                            self.TCN.append(DepthConv1d(BN_dim, H_dim, kernel, dilation=1, 
                                                        padding=1, skip=skip,
                                                        causal=causal, bool_drop=bool_drop, 
                                                        drop_value=drop_value, weight_norm=weight_norm))
                            self.dnn_layers.append(FiLMLayer(film_input, BN_dim)) 
                        else:
                            self.TCN.append(DepthConv1d(BN_dim, H_dim, kernel, dilation=2 * i,
                                                    padding=2 * i, skip=skip,
                                                    causal=causal, bool_drop=bool_drop, 
                                                    drop_value=drop_value, weight_norm=weight_norm)) ##I vhange to multiply and not square
                            self.dnn_layers.append(FiLMLayer(film_input, BN_dim))
                        if self.tf_attention:
                            self.time_freq_attnetion.append(TF_Attention())
                            
                    else:
                        self.TCN.append(
                            DepthConv1d(BN_dim, H_dim, kernel, dilation=1, padding=1, skip=skip, causal=causal, bool_drop=bool_drop,
                                         drop_value=drop_value, weight_norm=weight_norm))
                        self.dnn_layers.append(FiLMLayer(film_input, BN_dim))
                    if i == 0 and r == 0:
                        self.receptive_field += kernel
                    else:
                        if self.dilated:
                            self.receptive_field += (kernel - 1) * (i % 4 + 1)
                        else:
                            self.receptive_field += (kernel - 1)

            # output layer
            self.output = nn.Sequential(nn.PReLU(),
                                    nn.GroupNorm(1, BN_dim),
                                    nn.Conv1d(BN_dim, output_dim, 1)
                                    )
        else:
            if not causal:
                #this is like layer normalization because the number of groups is equal to one input_dim
                self.LN = nn.GroupNorm(1, input_dim, eps=1e-8) 
            else:
                self.LN = cLN(input_dim, eps=1e-8)

            # TCN for feature extraction
            self.receptive_field = 0
            self.dilated = dilated

            self.TCN = nn.ModuleList([])
            if self.tf_attention:
                self.time_freq_attnetion = nn.ModuleList([])
            for r in range(R):
                for i in range(layer):
                    if self.dilated:
                        if i == 0:
                            self.TCN.append(DepthConv1d(BN_dim, H_dim, kernel, dilation=1, padding=1, skip=skip,
                                                        causal=causal, bool_drop=bool_drop, drop_value=drop_value, weight_norm=weight_norm)) 
                            self.dnn_layers.append(FiLMLayer(film_input, BN_dim))
                        else:
                            self.TCN.append(DepthConv1d(BN_dim, H_dim, kernel, dilation=i%4+1, padding=i%4+1, skip=skip,
                                                    causal=causal, bool_drop=bool_drop, drop_value=drop_value, weight_norm=weight_norm))
                            self.dnn_layers.append(FiLMLayer(film_input, BN_dim))
                        if self.tf_attention:
                            self.time_freq_attnetion.append(TF_Attention())

                        
                    else:
                        self.TCN.append(
                            DepthConv1d(BN_dim, H_dim, kernel, dilation=1, padding=1, skip=skip, causal=causal, bool_drop=bool_drop,
                                        drop_value=drop_value, weight_norm=weight_norm))
                        self.dnn_layers.append(FiLMLayer(film_input, BN_dim))
                    if i == 0 and r == 0:
                        self.receptive_field += kernel
                    else:
                        if self.dilated:
                            self.receptive_field += (kernel - 1) * (i % 4 + 1)
                        else:
                            self.receptive_field += (kernel - 1)

            if self.apply_recursive_ln:
                self.ln_first_modules = nn.ModuleList([])
                self.ln_second_modules = nn.ModuleList([])
                for i in range(len(self.TCN)):
                    self.ln_first_modules.append(nn.GroupNorm(1, BN_dim))
                    self.ln_second_modules.append(nn.GroupNorm(1, BN_dim))
            if self.apply_residual_ln:
                self.ln_modules = nn.ModuleList([])
                for i in range(len(self.TCN)):
                    self.ln_modules.append(nn.GroupNorm(1, BN_dim))  
                    
            # output layer
            self.output = nn.Sequential(nn.PReLU(),
                                    nn.GroupNorm(1, BN_dim),
                                    wn(nn.Conv1d(BN_dim, output_dim, 1))
                                    )
            
                
    #@profile
    def forward(self, input, film_label):
        # input shape: (B, n_fft / 2, T)
        
        # normalization
        output = self.BN(self.LN(input)) ## Notice: There maybe some bug before, add self.BN part.
        # pass to TCN
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](output)
                gamma, beta = self.dnn_layers[i](film_label)
                residual = residual * gamma + beta
                output = output + residual
                skip_connection = skip_connection + skip
        else:

            for i in range(len(self.TCN)):
                residual = self.TCN[i](output)
                gamma, beta = self.dnn_layers[i](film_label)
                # print("shape: ", residual.shape, "gamma: ", gamma.shape, "beta: ", beta.shape)
                # shape:  torch.Size([10, 256, 188]) gamma:  torch.Size([10, 256]) beta:  torch.Size([10, 256])
                # multiple gamma and beta to each channel

                residual = residual * gamma + beta
                if self.tf_attention:
                    residual = self.time_freq_attnetion[i](residual)
                if self.apply_recursive_ln:
                    output = self.ln_second_modules[i](output + self.ln_first_modules[i](output + residual))
                elif self.apply_residual_ln:
                    output = output + self.ln_modules[i](residual)
                else:
                    output = output + residual
        # output layer
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output)
        return output


class F0ExtractModelFiLM(nn.Module):
    def __init__(self, **config):
        defaults  = {'n_fftBins': 512, 'BN_dim': 256, 'H_dim': 512, 'layer': 8, 'stack': 3, 'kernel': 3,
                            'num_spk': 1, 'skip': False, 'dilated': True, 'casual': False, 'bool_drop': True,
                            'drop_value': 0.1, 'weight_norm': False, 'final_vad': True, 'noisy_phase': False,
                            'activity_input_bool': False, 'tf_attention': False, 'apply_recursive_ln': False,
                            'apply_residual_ln': False, 'final_vad_masked_speakers': False, 'cate_num': 27,
                            'freq_label_num': 292}

        super(F0ExtractModelFiLM, self).__init__()
        
        defaults.update(config)
        print(defaults)
        # Set attributes from the dictionary.
        for key, value in defaults.items():
            setattr(self, key, value)  
        self.n_fftBins_h = self.n_fftBins // 2 + 1
        hop_length = int(self.n_fftBins/2) #32 self.
        input_tcn = self.n_fftBins_h - 1 #the one is the DC
        output_tcn = self.n_fftBins_h * self.num_spk

        self.enc = nn.Sequential(
            nn.Conv1d(in_channels=1,
                      out_channels=self.n_fftBins // 2, kernel_size=512, stride=256,
                      padding=0, bias=False),
            nn.ReLU())
        # conditional TCN with FiLM
        self.TCN = TCN_Film(input_tcn, output_tcn, self.BN_dim, self.H_dim,
                    self.layer, self.stack, kernel=3, skip=self.skip,
                    causal=self.casual, dilated=self.dilated, bool_drop=self.bool_drop, drop_value=self.drop_value, weight_norm=self.weight_norm,
                    tf_attention=self.tf_attention, apply_recursive_ln=self.apply_recursive_ln, apply_residual_ln=self.apply_residual_ln, film_input=self.cate_num)
                    
        self.m = nn.Sigmoid()
        # add condition layer: class to embedding 
        self.label_embedding = nn.Sequential(
            nn.Linear(self.cate_num, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, input_tcn),
            nn.LayerNorm(input_tcn),
            nn.ReLU())
        # final layer for F0 classification
        self.dnn_layers = nn.Sequential(
            nn.Linear(output_tcn, 512),
            nn.ReLU(),
            nn.Linear(512, self.freq_label_num)
        )

    def forward(self, x: torch.float32, cond_input: torch.float32, inference_kw={}):
        assert x.ndim == 2 , "input tensor must be 2 dimensions (B, T), but got dimensions of {}".format(x.ndim)            
        self.spectrum = self.enc(x.unsqueeze(1))
        if self.activity_input_bool:
            y = torch.unsqueeze(self.spectrum, dim=1)
            x = self.activity_input(y)
            spec_activity = torch.squeeze(x, dim=1)
            spec_activity = self.prelu(spec_activity)
            self.spectrum = self.spectrum * spec_activity
        self.masks_b = self.TCN(self.spectrum[:, :], cond_input) 
        f0_est = self.dnn_layers(self.masks_b.permute(0, 2, 1)).permute(0, 2, 1)
        return f0_est


if __name__ == "__main__":
    batch_size = 10
    class_label = 27 # category num
    seq_len = 16000 * 3
    freq_class = 292
    s = F0ExtractModelFiLM(cate_num=class_label, freq_label_num=freq_class)
    x = torch.rand(size=(batch_size, seq_len))
    emb = torch.rand(size=(batch_size, class_label)) 
    out_se = s(x, emb)
    print("done", out_se.shape) # [B, freq_label_num, frame_num] 48000/