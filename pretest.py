from os.path import join
import torch
from torch.utils.data import DataLoader

from utils import k_fold_split_train_val_test, RunningAverage
from model import patchPredictor
from datasets import patchPredictor_dataset

def main():
    for fold_num in [1,2,3,4,5]:
        # set directories
        root_dir = "/path/to/root/directory/"                           ## TODO: update path variable here ##
        source_dir = "/path/to/directory/containing/preprocessed/data/" ## TODO: update path variable here ##
        checkpoint_dir = join(root_dir, f"qaTool/models/patchPredictor/fold{fold_num}")
        ct_subvolume_dir = join(source_dir, "pretrain_ct_patches/")
        uniform_points_dir = join(source_dir, "pretrain_uniform_points/")

        # Create the model
        model = patchPredictor()
        model.load_best(checkpoint_dir=checkpoint_dir)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        
        # Create loss function
        LossFn = torch.nn.BCEWithLogitsLoss()

        # put the model on GPU(s)
        device='cuda'
        model.to(device)

        # Create dataloaders
        _, _, test_inds = k_fold_split_train_val_test(68, fold_num, seed=220469)
        test_data = patchPredictor_dataset(ct_subvolume_dir=ct_subvolume_dir, uniform_points_dir=uniform_points_dir, samples_per_epoch=8192, inds=test_inds, seed=220469)
        test_loader = DataLoader(dataset=test_data, batch_size=int(256), shuffle=False)
        
        # Test this model
        test_losses = RunningAverage()
        test_acc = RunningAverage()
        for batch_idx, sample in enumerate(test_loader):
            patch = sample['patch']
            label = sample['label']
            # forward pass
            with torch.cuda.amp.autocast():
                soft_pred = model(patch)
                # use BCE_loss
                loss = LossFn(soft_pred, label)
                # get accuracy
                acc = (torch.argmax(soft_pred.detach(), dim=1) == torch.argmax(label, dim=1)).sum() / label.size(0)
                # log
                test_losses.update(loss.item(), patch.size(0))
                test_acc.update(acc.item(), patch.size(0))

        # final results
        print(f"Fold {fold_num} model - test loss: {round(test_losses.avg, ndigits=4)} - test acc: {round(test_acc.avg, ndigits=4)}")
        '''
        # Original results
        Fold 1 model - test loss: 0.2441 - test acc: 0.8973
        Fold 2 model - test loss: 0.257 - test acc: 0.8898
        Fold 3 model - test loss: 0.2916 - test acc: 0.8757
        Fold 4 model - test loss: 0.2546 - test acc: 0.8916
        Fold 5 model - test loss: 0.297 - test acc: 0.8839
        '''

    # Romeo Dunn
    return

if __name__ == '__main__':
    main()