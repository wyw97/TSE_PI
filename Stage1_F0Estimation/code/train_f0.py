import torch 
from torch import optim
from pytorch_lightning import Trainer
from asteroid.models import DPRNNTasNet, DPTNet

from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper, singlesrc_neg_sisdr

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import os 
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn 
import numpy as np 

from system import SystemCondF0 as System
from model_f0_film import F0ExtractModelFiLM as Model

## TODO: Add the dataset class!!!

if __name__ == "__main__":
    # load dataset
    data_train, data_val = None, None
    # configuration
    save_folder = None 
    
    # check if load 
    assert data_train is not None and data_val is not None, "Please load the dataset first!"
    assert save_folder is not None, "Please specify the save folder!"
    for layer_num in [8]: # TCN layer
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        exp_dir = f"{save_folder}/layer_num_{layer_num}"
        use_cuda = True
        kwargs = {
            'num_workers': 8,
            'pin_memory': True
        } if use_cuda else {}
        train_loader = torch.utils.data.DataLoader(data_train,
                                                batch_size=64,
                                                shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(data_val,
                                                batch_size=64,
                                                **kwargs)
        model = Model(tf_attention=False, layer=layer_num)
        ckpt_pth = None
        if ckpt_pth is not None:
            model_state_dict = torch.load(ckpt_pth)['state_dict']
            model_state_dict = {k.replace("model.", ""): v for k, v in model_state_dict.items() if "model." in k}
            model.load_state_dict(model_state_dict)       
        logger = TensorBoardLogger("lightning_logs", name=f"{exp_dir}")
        os.makedirs(exp_dir, exist_ok=True)
        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        system = System(model, optimizer, loss, train_loader, val_loader)
        # Define callbacks
        callbacks = []
        checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
        checkpoint = ModelCheckpoint(
            checkpoint_dir, monitor="val_loss", mode="min", save_top_k=5, verbose=True
        )
        callbacks.append(checkpoint)
        # Train for 1 epoch using a single GPU. If you're running this on Google Colab,
        # be sure to select a GPU runtime (Runtime → Change runtime type → Hardware accelarator).
        trainer = Trainer(
            max_epochs=50,
            callbacks=callbacks,
            default_root_dir=exp_dir,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            strategy="ddp",
            devices="auto",
            gradient_clip_val=5,
            logger=logger,
        )
        trainer.fit(system)
