"""Layers related to pooling operations"""
# region
import typing as t

from torch import nn
import torch_scatter

from deep_scaffold import ops
# endregion


__all__ = ['AvgPooling', 'SumPooling']


class Pooling(nn.Module):
    """A abstract pooling layer"""
    def __init__(self,
                 in_features: int,
                 pooling_op: t.Callable = torch_scatter.scatter_mean,
                 activation: str = 'elu'):
        """
        Constructor

        Args:
            in_features (int):
                The number of input features
            pooling_op (Callable):
                The pooling operation, default to segment_mean
            activation (str):
                The type of activation function to use, default to elu

        """
        super(Pooling, self).__init__()
        self.bn_relu = nn.Sequential(nn.BatchNorm1d(in_features),
                                     ops.get_activation(activation,
                                                        inplace=True))
        self.pooling_op = pooling_op

    def forward(self, x, ids, num_seg=None):
        """
        Args:
            x (torch.Tensor): The input tensor, size=[N, in_features]
            ids (torch.Tensor): A tensor of type `torch.long`, size=[N, ]
            num_seg (int): The number of segments (graphs)

        Returns:
            torch.Tensor: Output tensor with size=[num_seg, in_features]
        """

        # performing batch_normalization and activation
        # size=[N, in_features]
        x_bn = self.bn_relu(x)

        # performing segment operation
        # size=[num_seg, in_features]
        x_pooled = self.pooling_op(x_bn,
                                   dim=0,
                                   index=ids,
                                   dim_size=num_seg)

        return x_pooled


class AvgPooling(Pooling):
    """Average pooling layer for graph"""
    def __init__(self,
                 in_features: int,
                 activation: str = 'elu'):
        """ Performing graph level average pooling (with bn_relu)

        Args:
            in_features (int):
                The number of input features
            activation (str):
                The type of activation function to use, default to elu
        """
        super(AvgPooling, self).__init__(in_features,
                                         activation=activation,
                                         pooling_op=torch_scatter.scatter_mean)


class SumPooling(Pooling):
    """Sum pooling layer for graph"""
    def __init__(self,
                 in_features: int,
                 activation: str = 'elu'):
        """ Performing graph level sum pooling (with bn_relu)

        Args:
            in_features (int):
                The number of input features
            activation (str):
                The type of activation function to use, default to elu
        """
        super(SumPooling, self).__init__(in_features,
                                         activation=activation,
                                         pooling_op=torch_scatter.scatter_add)
