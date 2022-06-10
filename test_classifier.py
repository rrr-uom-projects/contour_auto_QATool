import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import argparse as ap
import torch
from torch_geometric.loader import DataLoader
from utils import k_fold_split_train_val_test, str2bool, get_checkpoint_dir, ConfusionMatrix
from model import qaTool_classifier, qaTool_classifier_GNNAblation
from datasets import qaTool_classifier_dataset, qaTool_classifier_dataset_ablation
from tqdm import tqdm

def setup_argparse():
    parser = ap.ArgumentParser(prog="Main testing program for qaTool")
    parser.add_argument("--fold_num", choices=[1,2,3,4,5], type=int, help="The fold number for the kfold cross validation")
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

    # generate confusion matrix
    confusion_matrix = ConfusionMatrix(n_classes=5)

    for fold_num in [1,2,3,4,5]:
        # set directories
        root_dir = "/path/to/root/directory/"                           ## TODO: update path variable here ##
        source_dir = "/path/to/directory/containing/preprocessed/data/" ## TODO: update path variable here ##
        exp_dir, exp_name = get_checkpoint_dir(root_dir=join(root_dir, f"qaTool/models/classification/"), args=args)
        # add specific fold num
        checkpoint_dir = join(exp_dir, f"fold{fold_num}")
        # data directories
        mesh_dir = join(source_dir, "graph_objects/")
        gs_classes_dir = join(source_dir, "signed_classes/")
        ct_patches_dir = join(source_dir, "ct_patches/")
        triangles_dir= join(source_dir, "triangles_smooth/")
        # output directory
        preds_output_dir = join(checkpoint_dir, f"soft_preds/")
        try:
            os.mkdir(preds_output_dir)
        except OSError:
            pass
        
        # determine which meshes to use
        _, _, test_inds = k_fold_split_train_val_test(68, fold_num, seed=220469)

        # Create the models
        if args.GNNAbl:
            print("Processor ablation...")
            model = qaTool_classifier_GNNAblation(n_classes=5, processor=args.processor, spline_deg=args.spline_deg, kernel_size=args.kernel_size, aggr=args.aggr, mlp_features=args.decoder_feat)
        else:
            model = qaTool_classifier(n_classes=5, processor=args.processor, spline_deg=args.spline_deg, kernel_size=args.kernel_size, aggr=args.aggr, mlp_features=args.decoder_feat)
        model.load_best(checkpoint_dir)
        for param in model.parameters():
            param.requires_grad = False

        # put the model on GPU(s)
        device = 'cuda'
        model.to(device)
        model.eval()

        # Create dataloaders
        if args.encAbl:
            print("Encoder ablation...")
            test_data = qaTool_classifier_dataset_ablation(mesh_dir=mesh_dir, gs_classes_dir=gs_classes_dir, ct_patches_dir=ct_patches_dir, mesh_inds=test_inds, perturbations_to_sample_per_epoch=100)
        else:
            test_data = qaTool_classifier_dataset(mesh_dir=mesh_dir, gs_classes_dir=gs_classes_dir, ct_patches_dir=ct_patches_dir, mesh_inds=test_inds, perturbations_to_sample_per_epoch=100)
        test_loader = DataLoader(dataset=test_data, batch_size=int(1), shuffle=False)

        # Make predictions
        with torch.no_grad():
            for _, graph in enumerate(tqdm(test_loader)):
                fname = graph.fname[0]
                # send tensors to GPU
                graph = graph.to(device)

                # forward pass
                pred_node_classes = model(graph)
                
                # copy resulting node classes back to the cpu
                pred_node_classes = pred_node_classes.clone().detach().cpu().numpy()
        
                # save predictions
                np.save(join(preds_output_dir, f"{fname}.npy"), pred_node_classes)

                # confusion matrix update
                confusion_matrix.update(targets=graph.y.detach().cpu().numpy(), soft_preds=pred_node_classes)
                
    # generate confusion matrix figure
    fig = confusion_matrix.gen_matrix_fig()
    mat_dir = "/path/to/results/directory/"        ## TODO: update path variable here ##
    fig.savefig(join(mat_dir, f"{exp_name}.png"))
    confusion_matrix_data = confusion_matrix.retrieve_data()
    np.save(join(mat_dir, f"{exp_name}.npy"), confusion_matrix_data)

    # Romeo Dunn
    return

if __name__ == '__main__':
    main()