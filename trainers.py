## trainers.py
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import pyvista as pv
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from utils import save_checkpoint, RunningAverage, ConfusionMatrix, get_logger
import time
from kornia.losses import FocalLoss

#####################################################################################################
########################################### trainers ################################################
#####################################################################################################

class general_trainer:
    def __init__(self):
        self.epsilon = 1e-6

    def fit(self, verbose=True):
        self._save_init_state()
        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            t = time.time()
            should_terminate = self.train(self.train_loader)
            if verbose:
                print("Epoch trained in " + str(int(time.time()-t)) + " seconds.")
            if should_terminate:
                print("Hit termination condition...")
                break
            self.num_epoch += 1
        self.writer.close()
        return self.num_iterations, self.best_eval_score

    def _save_init_state(self):
        state = {'model_state_dict': self.model.state_dict()}
        init_state_path = os.path.join(self.checkpoint_dir, 'initial_state.pytorch')
        print(f"Saving initial state to '{init_state_path}'")
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        torch.save(state, init_state_path)

    def _is_best_eval_score(self, eval_score):
        is_best = eval_score > self.best_eval_score if self.eval_score_higher_is_better else eval_score < self.best_eval_score
        if is_best:
            self.logger.info(f'Saving new best evaluation metric: {eval_score}')
            self._log_new_best(eval_score)
            self.best_eval_score = eval_score
            self.epochs_since_improvement = 0
        return is_best

    def _save_checkpoint(self, is_best):
        save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': self.model.state_dict(),
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=self.logger)

    def _log_lr(self, new_lr):
        self.writer.add_scalar('learning_rate', new_lr, self.num_iterations)

    def _log_new_best(self, eval_score):
        self.writer.add_scalar('best_val_loss', eval_score, self.num_iterations)

    def _log_loss(self, phase, loss_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
        }
        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)
    
    def _log_acc(self, phase, acc_avg):
        tag_value = {
            f'{phase}_acc_avg': acc_avg,
        }
        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_epoch)

    def _log_params(self):
        self.logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)

#################################################
################ QA tool trainer ################
#################################################

class qaTool_classifier_trainer(general_trainer):
    def __init__(self, model, optimizer, lr_scheduler, device, train_loader, val_loader, logger, checkpoint_dir,
                class_weights, triangles_dir, max_num_epochs=100, patience=10, eval_score_higher_is_better=False):
        self.logger = logger
        self.logger.info(model)
        self.model = model
        self.optimizer = optimizer
        self.lr = self.optimizer.param_groups[0]['lr']
        self.scheduler = lr_scheduler
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.eval_score_higher_is_better = eval_score_higher_is_better
        # initialize the best_eval_score
        self.best_eval_score = float('-inf') if eval_score_higher_is_better else float('+inf')
        self.patience = patience
        self.epochs_since_improvement = 0
        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))
        self.num_epoch = 0
        self.num_iterations = 0
        #self.LossFn = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.LossFn = FocalLoss(alpha=class_weights, gamma=2., reduction='mean')
        self.triangles_dir = triangles_dir
        # pyvista virtual frame buffer
        pv.start_xvfb()
        # number of classes
        self.n_classes = class_weights.size(0)

    def train(self, train_loader):
        """Trains the model for 1 epoch.
        Args:
            train_loader (torch.utils.data.DataLoader): training data loader
        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = RunningAverage()
        improved = False        # for early stopping
        self.model.train()      # set the model in training mode
        for batch_idx, graph in enumerate(train_loader):       
            # send tensors to GPU
            graph = graph.to(self.device)

            # forward
            loss, _, _ = self._forward_pass(graph)
            train_losses.update(loss.item(), len(graph.indiv_num_nodes))
            
            # backward
            loss.backward()

            # optimizer
            self.optimizer.step()
            self.optimizer.zero_grad()

            # log stats
            if self.num_iterations % 20 == 0:
                self.logger.info(f'Training iteration {self.num_iterations}. Batch {batch_idx + 1}. Epoch [{self.num_epoch + 1}/{self.max_num_epochs}]')
                self.logger.info(f'Training stats. Loss: {train_losses.avg}')
            self._log_loss('train', train_losses.avg)
            self.num_iterations += 1
            
        # evaluate on validation set
        self.model.eval()
        eval_score = self.validate()

        # log current learning rate in tensorboard
        self._log_lr(self.lr)
        
        # adjust learning rate if necessary
        if self.scheduler is not None:
            self.scheduler.step()
        self.lr = self.optimizer.param_groups[0]['lr']

        # remember best validation metric
        improved = True if self._is_best_eval_score(eval_score) else False
        
        # save checkpoint
        self._save_checkpoint(improved)

        # implement early stopping here
        if not improved:
            self.epochs_since_improvement += 1
        if(self.epochs_since_improvement > self.patience):  # Model has not improved for certain number of epochs
            self.logger.info(
                    f'Model not improved for {self.patience} epochs. Finishing training...')
            return True
        return False    # Continue training...

    def validate(self):
        self.logger.info('Validating...')
        val_losses = RunningAverage()
        val_acc = RunningAverage()
        val_confusion_matrix = ConfusionMatrix(n_classes=self.n_classes)
        with torch.no_grad():
            for batch_idx, graph in enumerate(self.val_loader):
                if batch_idx % 10 == 0:
                    self.logger.info(f'Validation iteration {batch_idx + 1}')

                # send tensors to GPU
                graph = graph.to(self.device)
                 
                # run forward
                loss, acc, pred_node_classes = self._forward_pass(graph)
                
                # stats
                val_losses.update(loss.item(), len(graph.indiv_num_nodes))
                val_acc.update(acc.item(), sum(graph.indiv_num_nodes).item())
                
                # confusion matrix
                val_confusion_matrix.update(targets=graph.y.detach().cpu().numpy(), soft_preds=pred_node_classes.detach().cpu().numpy())

                # plot an example
                if batch_idx==0: 
                    self._plot_example(graph, pred_node_classes)
            # plot confusion matrix
            fig = val_confusion_matrix.gen_matrix_fig()
            self.writer.add_figure(tag='Confusion matrix', figure=fig, global_step=self.num_epoch)
            self._log_loss('val', val_losses.avg)
            self._log_acc('val', val_acc.avg)
            self.logger.info(f'Validation finished. Loss: {val_losses.avg} Accuracy: {val_acc.avg}')
            return val_losses.avg

    def _forward_pass(self, graph):
        # forward pass
        pred_node_classes = self.model(graph)
        # categorical cross-entropy loss
        loss = self.LossFn(pred_node_classes, graph.y)
        # calculate accuracy
        acc = (torch.argmax(pred_node_classes, dim=1)==graph.y).sum() / sum(graph.indiv_num_nodes).item()
        return loss, acc, pred_node_classes

    def _plot_example(self, graph, pred_node_classes):
        # setup
        fig, (ax_gs, ax_pred) = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
        axs = (ax_gs, ax_pred)
        # scrape an example mesh and set of node classes from the graph mini-batch
        indiv_num_nodes = graph.indiv_num_nodes
        num_nodes_to_get = indiv_num_nodes[0].item()
        node_coords = graph.pos[:num_nodes_to_get].detach().cpu().numpy()
        gs_classes = graph.y[:num_nodes_to_get].detach().cpu().numpy()
        pred_classes = torch.argmax(pred_node_classes, dim=1)[:num_nodes_to_get].detach().cpu().numpy()
        triangles = np.load(os.path.join(self.triangles_dir, f"{graph.fname[0]}.npy"))
        mesh = pv.PolyData(node_coords, np.insert(triangles, 0, 3, axis=1), deep=True, n_faces=triangles.shape[0])
        # use pyvista to render an example
        pv.global_theme.background = 'white'
        plotter_gs = pv.Plotter(off_screen=True)
        plotter_pred = pv.Plotter(off_screen=True)
        plotter_gs.add_mesh(mesh, color=[1, 0.706, 0], opacity=1)
        plotter_pred.add_mesh(mesh, color=[1, 0.706, 0], opacity=1)
        cmap = plt.cm.get_cmap('bwr')
        for pc_idx in range(self.n_classes):
            if (pc_idx==gs_classes).any():
                pc = pv.PolyData(node_coords[pc_idx==gs_classes])
                plotter_gs.add_mesh(pc, render_points_as_spheres=True, color=cmap(pc_idx/(self.n_classes-1))[:3], point_size=15)
            if (pc_idx==pred_classes).any():
                pc = pv.PolyData(node_coords[pc_idx==pred_classes])
                plotter_pred.add_mesh(pc, render_points_as_spheres=True, color=cmap(pc_idx/(self.n_classes-1))[:3], point_size=15)
        plotter_gs.camera.roll, plotter_gs.camera.elevation, plotter_gs.camera.azimuth = 120, 0, -120
        plotter_pred.camera.roll, plotter_pred.camera.elevation, plotter_pred.camera.azimuth = 120, 0, -120
        plotter_gs.store_image, plotter_pred.store_image = True, True
        plotter_gs.show(window_size=[500, 500], auto_close=True)
        plotter_pred.show(window_size=[500, 500], auto_close=True)
        # jump to matplotlib
        ax_gs.imshow(plotter_gs.image)
        ax_pred.imshow(plotter_pred.image)
        ax_gs.axis('off')
        ax_pred.axis('off')
        ax_gs.set_title(f"Gold standard")
        ax_pred.set_title(f"Prediction")
        self.writer.add_figure(tag='Class_preds', figure=fig, global_step=self.num_epoch)

#################################################
############ patchPredictor trainer #############
#################################################

class patchPredictor_trainer(general_trainer):
    def __init__(self, model, optimizer, lr_scheduler, device, train_loader, val_loader,
                checkpoint_dir, max_num_epochs=100, patience=10, eval_score_higher_is_better=False):
        self.logger = get_logger('pP_Training')
        self.model = model
        self.optimizer = optimizer
        self.lr = self.optimizer.param_groups[0]['lr']
        self.scheduler = lr_scheduler
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.eval_score_higher_is_better = eval_score_higher_is_better
        # initialize the best_eval_score
        self.best_eval_score = float('-inf') if eval_score_higher_is_better else float('+inf')
        self.patience = patience
        self.epochs_since_improvement = 0
        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))
        self.num_epoch = 0
        self.num_iterations = 0
        self.scaler = torch.cuda.amp.GradScaler()
        self.LossFn = torch.nn.BCEWithLogitsLoss()

    def train(self, train_loader):
        """Trains the model for 1 epoch.
        Args:
            train_loader (torch.utils.data.DataLoader): training data loader
        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = RunningAverage()
        train_acc = RunningAverage()
        improved = False        # for early stopping
        self.model.train()      # set the model in training mode
        for batch_idx, sample in enumerate(train_loader):
            patch = sample['patch']
            label = sample['label']

            # forward
            loss, acc, _ = self._forward_pass(patch, label)
            train_losses.update(loss.item(), patch.size(0))
            train_acc.update(acc.item(), patch.size(0))
            
            # compute gradients and update parameters
            # Native AMP training step
            self.scaler.scale(loss).backward()
            
            # call step() and reset gradients:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            # log stats
            self._log_loss('train', train_losses.avg)
            self._log_acc('train',  train_acc.avg)
            self.num_iterations += 1

        # log to terminal
        print(f'Epoch [{self.num_epoch + 1}/{self.max_num_epochs}] training stats: Loss: {train_losses.avg} Accuracy: {train_acc.avg}')

        # evaluate on validation set
        self.model.eval()
        eval_score = self.validate()
            
        # log current learning rate in tensorboard
        self._log_lr(self.lr)

        # adjust learning rate if necessary
        if self.scheduler is not None:
            self.scheduler.step()
        self.lr = self.optimizer.param_groups[0]['lr']

        # remember best validation metric
        improved = True if self._is_best_eval_score(eval_score) else False
        
        # save checkpoint
        self._save_checkpoint(improved)

        # implement early stopping here
        if not improved:
            self.epochs_since_improvement += 1
        if(self.epochs_since_improvement > self.patience):  # Model has not improved for certain number of epochs
            print(f'Model not improved for {self.patience} epochs. Finishing training...')
            return True
        return False    # Continue training...
        

    def validate(self):
        val_losses = RunningAverage()
        val_acc = RunningAverage()
        with torch.no_grad():
            which_to_show = np.random.randint(0, self.val_loader.batch_size)    # show a random example from a batch
            for batch_idx, sample in enumerate(self.val_loader):
                patch = sample['patch']
                label = sample['label']
                
                loss, acc, soft_pred = self._forward_pass(patch, label)
                val_losses.update(loss.item(), patch.size(0))
                val_acc.update(acc.item(), patch.size(0))
                
            self._log_loss('val', val_losses.avg)
            self._log_acc('val',  val_acc.avg)
            print(f'Validation finished. Loss: {val_losses.avg} Accuracy: {val_acc.avg}')
            self.plot_example(patch[0,0].clone().detach().cpu().numpy(), label[0].clone().detach().cpu().numpy(), soft_pred[0].clone().detach().cpu().numpy())
            return val_losses.avg

    def _forward_pass(self, patch, label):
        with torch.cuda.amp.autocast():
            # forward pass
            soft_pred = self.model(patch)
            # use BCE_loss
            loss = self.LossFn(soft_pred, label)
            # get accuracy
            acc = (torch.argmax(soft_pred.detach(), dim=1) == torch.argmax(label, dim=1)).sum() / label.size(0)
            return loss, acc, soft_pred

    def plot_example(self, patch, label, soft_pred):
        fig, (ax0,ax1,ax2,ax3,ax4) = plt.subplots(1, 5, figsize=(15, 3), tight_layout=True)
        axs = (ax0,ax1,ax2,ax3,ax4)
        for ax_idx, ax in enumerate(axs):
            ax.imshow(patch[ax_idx].astype(float), cmap='Greys_r', vmin=0, vmax=1)
        label = "On contour" if np.argmax(label) == 0 else "Not on contour"
        pred = "On contour" if np.argmax(soft_pred) == 0 else "Not on contour"
        ax2.set_title(f"Label: {label}, Pred: {pred}")
        self.writer.add_figure(tag='Patch_pred', figure=fig, global_step=self.num_epoch)
