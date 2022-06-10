from os.path import join
import argparse as ap

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, StepLR, CosineAnnealingLR
from torch_geometric.loader import DataLoader

from utils import get_logger, get_number_of_learnable_parameters, k_fold_split_train_val_test, str2bool, get_checkpoint_dir
from model import qaTool_classifier, qaTool_classifier_GNNAblation
from trainers import qaTool_classifier_trainer
from datasets import qaTool_classifier_dataset, qaTool_classifier_dataset_ablation


def setup_argparse():
    parser = ap.ArgumentParser(prog="Main training program for qaTool")
    parser.add_argument("--fold_num", choices=[1,2,3,4,5], type=int, help="The fold number for the kfold cross validation")
    parser.add_argument("--use_pretrained_CNN", default=True, type=lambda x: bool(str2bool(x)), help="Whether to use the self-supervised CNN pretrained model")
    parser.add_argument("--lock_pretrained_CNN", default=False, type=lambda x: bool(str2bool(x)), help="Whether to lock the self-supervised CNN pretrained model")
    parser.add_argument("--init_lr", default=0.005, type=float, help="The initial lr")
    parser.add_argument("--processor", default="spline", type=str, help="The type of processor to use")
    parser.add_argument("--decoder_feat", default=128, type=int)
    parser.add_argument("--decoder_bn", default=True, type=lambda x: bool(str2bool(x)))
    parser.add_argument("--spline_deg", default=1, type=int)
    parser.add_argument("--kernel_size", default=5, type=int)
    parser.add_argument("--aggr", default="add", type=str)
    parser.add_argument("--lr_sched", default="expS", type=str)
    parser.add_argument("--bs", default=16, type=int)
    parser.add_argument("--dtspe", default=25, type=int)
    parser.add_argument("--dropout", default=True, type=lambda x: bool(str2bool(x)))
    parser.add_argument("--encAbl", default=False, type=lambda x: bool(str2bool(x)))
    parser.add_argument("--GNNAbl", default=False, type=lambda x: bool(str2bool(x)))
    parser.add_argument("--preAbl", default=False, type=lambda x: bool(str2bool(x)))
    global args
    args = parser.parse_args()

def main():
    # get args
    setup_argparse()
    global args

    # set directories
    root_dir = "/path/to/root/directory/"                           ## TODO: update path variable here ##
    source_dir = "/path/to/directory/containing/preprocessed/data/" ## TODO: update path variable here ##
    checkpoint_dir, _ = get_checkpoint_dir(root_dir=join(root_dir, f"qaTool/models/classification/"), args=args)
    # add specific fold num
    checkpoint_dir = join(checkpoint_dir, f"fold{args.fold_num}")
    mesh_dir = join(source_dir, "graph_objects/")
    gs_classes_dir = join(source_dir, "signed_classes/")
    ct_patches_dir = join(source_dir, "ct_patches/")
    triangles_dir= join(source_dir, "triangles_smooth/")

    # determine which meshes to use
    train_inds, val_inds, _ = k_fold_split_train_val_test(68, args.fold_num, seed=220469)

    # set device
    device = 'cuda'

    # Create logger
    logger = get_logger('iI_Training')

    # load and calculate the class weights
    class_counts = torch.load(join(source_dir, "all_signed_classes.pt"))
    inv_class_counts = 1 / class_counts
    class_weights = inv_class_counts / torch.sum(inv_class_counts)
    class_weights = class_weights.to(device)

    # Create the models
    n_classes = class_counts.size(0)
    if args.GNNAbl:
        print("Processor ablation...")
        model = qaTool_classifier_GNNAblation(n_classes=n_classes, processor=args.processor, spline_deg=args.spline_deg, kernel_size=args.kernel_size, aggr=args.aggr, mlp_features=args.decoder_feat)
    else:
        model = qaTool_classifier(n_classes=n_classes, processor=args.processor, spline_deg=args.spline_deg, kernel_size=args.kernel_size, aggr=args.aggr, mlp_features=args.decoder_feat)
    # load pretraining weights
    if not args.preAbl:
        model.load_pretrained_CNN(weights_path=join(root_dir, f"qaTool/models/patchPredictor/fold{args.fold_num}/best_checkpoint.pytorch"), logger=logger)
    
    # set specific bits to train
    for param in model.parameters():
        param.requires_grad = True

    # put the models on GPU(s)
    model.to(device)

    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')
    
    # Batch size with fancy graph batching
    train_BS = int(args.bs)
    val_BS = int(args.bs)

    # Create dataloaders
    if args.encAbl:
        print("Encoder ablation...")
        train_data = qaTool_classifier_dataset_ablation(mesh_dir=mesh_dir, gs_classes_dir=gs_classes_dir, ct_patches_dir=ct_patches_dir, mesh_inds=train_inds, perturbations_to_sample_per_epoch=args.dtspe)
        val_data = qaTool_classifier_dataset_ablation(mesh_dir=mesh_dir, gs_classes_dir=gs_classes_dir, ct_patches_dir=ct_patches_dir, mesh_inds=val_inds, perturbations_to_sample_per_epoch=100)
    else:    
        train_data = qaTool_classifier_dataset(mesh_dir=mesh_dir, gs_classes_dir=gs_classes_dir, ct_patches_dir=ct_patches_dir, mesh_inds=train_inds, perturbations_to_sample_per_epoch=args.dtspe)
        val_data = qaTool_classifier_dataset(mesh_dir=mesh_dir, gs_classes_dir=gs_classes_dir, ct_patches_dir=ct_patches_dir, mesh_inds=val_inds, perturbations_to_sample_per_epoch=100)    
    train_loader = DataLoader(dataset=train_data, batch_size=train_BS, num_workers=8, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=val_BS, num_workers=8, shuffle=True)

    # Create the optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.init_lr, weight_decay=0.001)

    # Create learning rate adjustment strategy
    if args.lr_sched == "noS":
        lr_scheduler = None
    elif args.lr_sched == "expS":
        lr_scheduler = ExponentialLR(optimizer, gamma=0.95)
    elif args.lr_sched == "cosS":
        lr_scheduler  = CosineAnnealingLR(optimizer, T_max=50)
    
    # Create model trainer
    trainer = qaTool_classifier_trainer(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, device=device, train_loader=train_loader, triangles_dir=triangles_dir,
                                        val_loader=val_loader, logger=logger, checkpoint_dir=checkpoint_dir, patience=30, max_num_epochs=50, class_weights=class_weights)
    
    # Start training
    trainer.fit()

    # Romeo Dunn
    return

if __name__ == '__main__':
    main()