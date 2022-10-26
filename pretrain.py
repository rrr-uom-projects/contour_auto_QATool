import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import argparse as ap

from utils import get_logger, get_number_of_learnable_parameters, k_fold_split_train_val_test
from model import patchPredictor
from trainers import patchPredictor_trainer
from datasets import patchPredictor_dataset

def setup_argparse():
    parser = ap.ArgumentParser(prog="Main training program for patchPredictor")
    parser.add_argument("--fold_num", choices=[1,2,3,4,5], type=int, help="The fold number for the kfold cross validation")
    global args
    args = parser.parse_args()

def main():
    # get args
    setup_argparse()
    global args

    # set directories
    root_dir = "/path/to/root/directory/"                           ## TODO: update path variable here ##
    source_dir = "/path/to/directory/containing/preprocessed/data/" ## TODO: update path variable here ##
    try_mkdir(join(root_dir, "models/"))
    models_dir =  join(root_dir, "models/patchPredictor/")
    try_mkdir(models_dir)
    checkpoint_dir = join(models_dir, f"fold{args.fold_num}/")
    ct_subvolume_dir = join(source_dir, "pretrain_ct_patches/")
    uniform_points_dir = join(source_dir, "pretrain_uniform_points/")

    # Create the model
    model = patchPredictor()

    for param in model.parameters():
        param.requires_grad = True

    # put the model on GPU(s)
    device='cuda'
    model.to(device)

    # Log the number of learnable parameters
    print(f'Number of learnable params {get_number_of_learnable_parameters(model)}')
    train_BS = int(64)
    val_BS = int(32)

    # Create dataloaders
    train_inds, val_inds, _ = k_fold_split_train_val_test(68, args.fold_num, seed=220469)
    train_data = patchPredictor_dataset(ct_subvolume_dir=ct_subvolume_dir, uniform_points_dir=uniform_points_dir, samples_per_epoch=512, inds=train_inds)
    train_loader = DataLoader(dataset=train_data, batch_size=train_BS, shuffle=True)
    val_data = patchPredictor_dataset(ct_subvolume_dir=ct_subvolume_dir, uniform_points_dir=uniform_points_dir, samples_per_epoch=128, inds=val_inds)
    val_loader = DataLoader(dataset=val_data, batch_size=val_BS, shuffle=True)

    # Create the optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=0.001)

    # Create learning rate adjustment strategy
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=10000)
    
    # Create model trainer
    trainer = patchPredictor_trainer(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, device=device, train_loader=train_loader,  
                                    val_loader=val_loader, checkpoint_dir=checkpoint_dir, patience=7500, max_num_epochs=10000)
    
    # Start training
    trainer.fit(verbose=False)

    # Romeo Dunn
    return

if __name__ == '__main__':
    main()