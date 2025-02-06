"""
@Email: yiting.chen@rice.edu
"""
import copy
import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict


class ResBlock(nn.Module):
    """
    Residual Block
    """
    def __init__(self, dim):
        super(ResBlock, self).__init__()

        self.fc_1 = nn.Linear(dim, round(dim / 2))
        self.fc_2 = nn.Linear(round(dim / 2), dim)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.fc_1(x)
        out = self.act(out)

        out = self.fc_2(out)
        out += residual
        out = self.act(out)
        return out

class ResBackbone(nn.Module):
    """
    Backbone with Residual Connection
    """
    def __init__(self, input_c=10, N=64, dropout_rate=0.2):
        super(ResBackbone, self).__init__()
        self.fc_1 = nn.Linear(3 * input_c, N)
        self.fc_2 = nn.Linear(N, round(N / 2))
        self.fc_3 = nn.Linear(round(N / 2), round(N / 4))
        self.fc_4 = nn.Linear(round(N / 4), round(N / 2))
        self.fc_5 = nn.Linear(round(N / 2), N - 3 * input_c)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # position encoding
        x_skip = torch.cat((x, torch.sin(x), torch.cos(x)), dim=-1)
        x = self.fc_1(x_skip)
        x = self.act(x)

        x = self.dropout(x)
        x = self.fc_2(x)
        x = self.act(x)

        x = self.dropout(x)
        x = self.fc_3(x)
        x = self.act(x)

        x = self.dropout(x)
        x = self.fc_4(x)
        x = self.act(x)

        # skip connection
        x = self.dropout(x)
        x = self.fc_5(x)
        x = self.act(x)
        x = torch.cat((x, x_skip), dim=1)

        return x


class MLP_Block(nn.Module):
    """
    Multi-layer Perceptron block with intermediate features
    """
    def __init__(self, input_c, output_c, h_layers=None):
        super(MLP_Block, self).__init__()
        if h_layers is None:
            h_layers = [64, 32, 16, 8]
        self.h_layers = copy.deepcopy(h_layers)
        self.h_layers.insert(0, input_c)
        self.h_layers.append(output_c)

        layers = OrderedDict()
        num_layers = len(self.h_layers)
        for i in range(num_layers - 1):
            layers['linear_{}'.format(i)] = nn.Linear(self.h_layers[i], self.h_layers[i + 1])
            if i < num_layers - 2:
                layers['gelu_{}'.format(i)] = nn.GELU()

        self.resBlock = ResBlock(input_c)
        self.model = nn.Sequential(layers)

    def forward(self, x, feat_layers=None, fused_feats=None, fused_layers=None):
        x = self.resBlock(x)

        # feat_layers = [1, 3, 5, 7]
        if (feat_layers is not None) and (fused_feats is None):
            feat = x
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in feat_layers:
                    feats.append(feat)
            return feat, feats
        elif (feat_layers is not None) and (fused_feats is not None) and (fused_layers is not None):
            assert len(fused_feats) == len(fused_layers)
            feat = x
            feats = []
            fused_id = 0
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in feat_layers:
                    feat = torch.add(feat, fused_feats[fused_id])
                    fused_id += 1
                if layer_id in feat_layers:
                    feats.append(feat)
            return feat, feats
        else:
            return self.model(x), None


class RobotNDF(nn.Module):
    """
    Robot Neural Distance Function
    input_c: nDoF(7) + xyz(3)
    N: network scale
    start_layer_index, end_layer_index: layers for feature concatenation
    nDoF: number of degree of freedoms
    """
    def __init__(self, input_c=10, N=64, start_layer_index=1, end_layer_index=2, dropout_rate=0.2):
        super(RobotNDF, self).__init__()
        # layer features
        self.start_index = start_layer_index
        self.end_index = end_layer_index
        # number of DoFs
        self.nDoF = input_c - 3
        assert N > (3 * input_c + 10)
        self.backbone = ResBackbone(input_c=input_c, N=N, dropout_rate=dropout_rate)
        for i in range(self.nDoF+1):
            setattr(self, "mlp_{}".format(i), MLP_Block(input_c=N, output_c=1))

    def forward(self, x):
        layers = np.arange(len(self.mlp_0.h_layers[self.start_index: self.end_index]))
        layers = 2*layers + 1

        x = self.backbone(x)
        pred_x = []
        feats = None
        for i in range(self.nDoF+1):
            if i == 0:
                pred, feats = getattr(self, "mlp_{}".format(i))(x, feat_layers=layers)
                pred_x.append(pred)
            else:
                pred, feats = getattr(self, "mlp_{}".format(i))(x, feat_layers=layers, fused_feats=feats, fused_layers=layers)
                pred_x.append(pred)

        pred_x = torch.cat(pred_x, dim=1)
        return pred_x * -1

    def load_model(self, load_path):
        state_dict = torch.load(load_path, weights_only=True)
        self.load_state_dict(state_dict, strict=True)


class Identity(nn.Module):
    def forward(self, x):
        return x

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.eps = eps

    def forward(self, x, y):
        loss = torch.sqrt(self.mse(x, y) + self.eps)
        return loss