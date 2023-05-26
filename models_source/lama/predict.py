#!/usr/bin/env python3

# Example command:
# ./bin/predict.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>
import os

from .saicinpainting.evaluation.utils import move_to_device

import cv2
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from .saicinpainting.training.data.datasets import make_default_val_dataset
from .saicinpainting.training.trainers import load_checkpoint
from .saicinpainting.utils import register_debug_signal_handlers

def imageinpainting_lama(indir, model, checkpoint, out_path):
    try:
        register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        device = torch.device('cuda')

        train_config_path = model
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        checkpoint_path = checkpoint
        
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()

        model.to(device)

        if not indir.endswith('/'):
            indir += '/'

        dataset = make_default_val_dataset(indir, **{'kind': 'default', 'img_suffix': '.png', 'pad_out_to_modulo': 8})
        print(dataset)
        for img_i in tqdm.trange(len(dataset)):
            cur_out_fname = out_path
            os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
            batch = default_collate([dataset[img_i]])
            
            with torch.no_grad():
                batch = move_to_device(batch, device)
                batch['mask'] = (batch['mask'] > 0) * 1
                batch = model(batch)                    
                cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
                unpad_to_size = batch.get('unpad_to_size', None)
                if unpad_to_size is not None:
                    orig_height, orig_width = unpad_to_size
                    cur_res = cur_res[:orig_height, :orig_width]

            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            cv2.imwrite(cur_out_fname, cur_res)

    except Exception as ex:
        print(ex)

