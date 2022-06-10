import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SplineConv, GATv2Conv

from collections import OrderedDict

class general_model(nn.Module):
    # template class so all models inherit the load_best method
    def __init__(self):
        super(general_model, self).__init__()

    def load_best(self, checkpoint_dir, logger=None):
        # load previous best weights
        model_dict = self.state_dict()
        state = torch.load(os.path.join(checkpoint_dir, 'best_checkpoint.pytorch'))
        best_checkpoint_dict = state['model_state_dict']
        # remove the 'module.' wrapper
        renamed_dict = OrderedDict()
        for key, value in best_checkpoint_dict.items():
            new_key = key.replace('module.','')
            renamed_dict[new_key] = value
        # identify which layers to grab
        renamed_dict = {k: v for k, v in list(renamed_dict.items()) if k in model_dict}
        model_dict.update(renamed_dict)
        self.load_state_dict(model_dict)
        if logger:
            logger.info("Loaded layers from previous best checkpoint:")
            logger.info([k for k, v in list(renamed_dict.items())])
        else:
            print("Loaded layers from previous best checkpoint:")
            print([k for k, v in list(renamed_dict.items())])
    
    def unlock_all(self):
        print("Unlocking all!")
        for module in self.children():
            for param in module.parameters():
                param.requires_grad = True

###################################################
########### node classification models ############
###################################################

class qaTool_classifier(general_model):
    def __init__(self, n_classes, device="cuda", processor="spline", spline_deg=2, kernel_size=5, aggr="mean", mlp_features=128):
        super(qaTool_classifier, self).__init__()
        # define the CNN patch encoder
        self.encoder = patchPredictor().encoder
        self.pooling = nn.AdaptiveAvgPool3d(1)

        # define the GNN processor
        if processor=="spline":
            self.processor = SplineProcessor(spline_deg, kernel_size, aggr)
        elif processor=="GAT":
            self.processor = GATProcessor()
        else:
            raise NotImplementedError

        # define the MLP decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features=32, out_features=mlp_features),
            nn.Dropout(p=0.25, inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(num_features=mlp_features),
            nn.Linear(in_features=mlp_features, out_features=mlp_features),
            nn.Dropout(p=0.25, inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(num_features=mlp_features),
            nn.Linear(in_features=mlp_features, out_features=n_classes)
        )
        self.device = device

    def forward(self, graph):
        # first encode the node patches - patches tensor of shape (batch*n_nodes, 5, 5, 5)
        # add channels dim
        graph.x = torch.squeeze(self.pooling(self.encoder(torch.unsqueeze(graph.patches_tensor, dim=1))))
                
        # patch embeddings now of shape (batch_size * n_nodes, 32) --> 32 arbitrary feature tensor size -> tune
        # now use the processor to perform geometric learning baby
        node_embeddings = self.processor(graph.x, graph.edge_index, graph.edge_attr)

        # now use the MLP decoder to predict error for each node
        out = self.decoder(node_embeddings)
        return out

    def load_pretrained_CNN(self, weights_path, logger):
        # load weights of the selected pretrained CNN
        model_dict = self.state_dict()
        state = torch.load(weights_path)
        pretrained_dict = state['model_state_dict']
        # identify which layers to grab
        pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict}
        # update the contourCorrectors weights
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        # log which layers have been altered
        logger.info("Loaded layers from pretrained CNN:")
        logger.info([k for k, v in list(pretrained_dict.items())])

    def lock_pretrained_CNN(self):
        for module_name, module in self.named_children():
            if module_name == "encoder":
                for name, param in module.named_parameters():
                    print(f"Locking {name}")
                    param.requires_grad = False

class qaTool_classifier_GNNAblation(general_model):
    def __init__(self, n_classes, device="cuda", processor="spline", spline_deg=2, kernel_size=5, aggr="mean", mlp_features=128):
        super(qaTool_classifier_GNNAblation, self).__init__()
        # define the CNN patch encoder
        self.encoder = patchPredictor().encoder
        self.pooling = nn.AdaptiveAvgPool3d(1)

        # define the GNN processor
        self.processor = None
        
        # define the MLP decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features=32, out_features=mlp_features),
            nn.Dropout(p=0.25, inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(num_features=mlp_features),
            nn.Linear(in_features=mlp_features, out_features=mlp_features),
            nn.Dropout(p=0.25, inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(num_features=mlp_features),
            nn.Linear(in_features=mlp_features, out_features=n_classes)
        )
        self.device = device

    def forward(self, graph):
        # first encode the node patches - patches tensor of shape (batch*n_nodes, 5, 5, 5)
        # add channels dim
        node_embeddings = torch.squeeze(self.pooling(self.encoder(torch.unsqueeze(graph.patches_tensor, dim=1))))
                
        # skip the processor!

        # now use the MLP decoder to predict error for each node
        out = self.decoder(node_embeddings)
        return out
    
    def load_pretrained_CNN(self, weights_path, logger):
        # load weights of the selected pretrained CNN
        model_dict = self.state_dict()
        state = torch.load(weights_path)
        pretrained_dict = state['model_state_dict']
        # identify which layers to grab
        pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict}
        # update the contourCorrectors weights
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        # log which layers have been altered
        logger.info("Loaded layers from pretrained CNN:")
        logger.info([k for k, v in list(pretrained_dict.items())])

######################################
############ Processors ##############
######################################

class SplineProcessor(general_model):
    def __init__(self, spline_deg, kernel_size, aggr):
        super(SplineProcessor, self).__init__()
        in_channels=32
        out_channels=32
        hidden_channels=32
        
        self.conv1 = SplineConv(in_channels=in_channels, out_channels=hidden_channels, dim=3, kernel_size=kernel_size, degree=spline_deg, aggr=aggr)
        self.nonlin1 = nn.LeakyReLU()
        self.norm1 = nn.BatchNorm1d(num_features=hidden_channels)

        self.conv2 = SplineConv(in_channels=hidden_channels, out_channels=hidden_channels, dim=3, kernel_size=kernel_size, degree=spline_deg, aggr=aggr)
        self.nonlin2 = nn.LeakyReLU()
        self.norm3 = nn.BatchNorm1d(num_features=hidden_channels)
        
        self.conv3 = SplineConv(in_channels=hidden_channels, out_channels=out_channels, dim=3, kernel_size=kernel_size, degree=spline_deg, aggr=aggr)
        self.norm5 = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, patch_embs, edge_index, edge_attr):
        ## advanced minibatching here:
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html          
        
        # using simple 3 layer GCN with residual connections
        x = self.norm1(self.nonlin1(self.conv1(patch_embs, edge_index, edge_attr)) + patch_embs)
        x = self.norm3(self.nonlin2(self.conv2(x, edge_index, edge_attr)) + x)
        x = self.norm5(self.conv3(x, edge_index, edge_attr) + x)
        return x

class GATProcessor(general_model):
    def __init__(self):
        super(GATProcessor, self).__init__()
        in_channels=32
        out_channels=32
        hidden_channels=32
        
        self.conv1 = GATv2Conv(in_channels=in_channels, out_channels=hidden_channels, heads=8, concat=False)
        self.nonlin1 = nn.LeakyReLU()
        self.norm1 = nn.BatchNorm1d(num_features=hidden_channels)

        self.conv2 = GATv2Conv(in_channels=in_channels, out_channels=hidden_channels, heads=8, concat=False)
        self.nonlin2 = nn.LeakyReLU()
        self.norm3 = nn.BatchNorm1d(num_features=hidden_channels)
        
        self.conv3 = GATv2Conv(in_channels=in_channels, out_channels=hidden_channels, heads=8, concat=False)
        self.norm5 = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, patch_embs, edge_index, edge_attr):
        ## advanced minibatching here:
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html          
        
        # using simple 3 layer GCN with residual connections
        x = self.norm1(self.nonlin1(self.conv1(patch_embs, edge_index)) + patch_embs)
        x = self.norm3(self.nonlin2(self.conv2(x, edge_index)) + x)
        x = self.norm5(self.conv3(x, edge_index) + x)
        return x


###################################################
######## self-supervised pretraining model ########
###################################################


class patchPredictor(general_model):
    def __init__(self):
        super(patchPredictor, self).__init__()
        # define the CNN patch encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(inplace=True)
        )
        self.pred = nn.Conv3d(in_channels=32, out_channels=2, kernel_size=1)
        self.pooling = nn.AdaptiveAvgPool3d(1)

    def forward(self, patch):
        # encode the patch
        out = self.encoder(patch)
        # pass it through a prediction layer
        out = self.pred(out)
        # apply GAP
        out = self.pooling(out).squeeze()
        # Logits applied in lossFn
        return out