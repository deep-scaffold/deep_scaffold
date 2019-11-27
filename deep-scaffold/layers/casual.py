"""Implements casual molecule convlutional block"""
# region
import typing as t

import torch
from torch import nn

from deep_scaffold.layers.mol_conv import MolConv
# endregion


__all__ = ['CausalMolConvBlock']


class CausalMolConvBlock(nn.Module):
    """The causal mol conv block"""

    def __init__(self,
                 num_atom_features: int,
                 num_bond_types: int,
                 hidden_sizes: t.Iterable,
                 activation: str = 'elu',
                 conditional: bool = False,
                 num_cond_features: t.Optional[int] = None,
                 activation_cond: t.Optional[str] = None):
        """ The constructor

        Args:
            num_atom_features (int):
                The number of input features for each node
            num_bond_types (int):
                The number of bond types considered
            hidden_sizes (t.Iterable):
                The hidden size and output size for each weave layer
            activation (str):
                The type of activation unit to use in this module,
                default to elu
            conditional (bool):
                Whether to include conditional input, default to False
            num_cond_features (int):
                The size of conditional input, should be None if
                self.conditional is False
            activation_cond (str or None):
                activation function used for conditional input,
                should be None if self.conditional is False
        """
        super(CausalMolConvBlock, self).__init__()

        self.num_node_features = num_atom_features
        self.num_bond_types = num_bond_types
        self.hidden_sizes = list(hidden_sizes)
        self.activation = activation
        self.conditional = conditional
        self.num_cond_features = num_cond_features
        self.activation_cond = activation_cond

        layers = []
        for i, (in_features, out_features) in \
                enumerate(zip([self.num_node_features, ] +
                              list(self.hidden_sizes)[:-1],  # in_features
                              self.hidden_sizes)):  # out_features
            if i == 0:
                layers.append(MolConv(in_features,
                                      self.num_bond_types,
                                      out_features,
                                      None,
                                      self.conditional,
                                      self.num_cond_features,
                                      self.activation_cond))
            else:
                layers.append(MolConv(in_features,
                                      self.num_bond_types,
                                      out_features,
                                      self.activation,
                                      self.conditional,
                                      self.num_cond_features,
                                      self.activation_cond))

        self.layers = nn.ModuleList(layers)

    def forward(self,
                atom_features: torch.Tensor,
                bond_info: torch.Tensor,
                cond_features: t.Optional[torch.Tensor] = None):
        """
        Args:
            atom_features (torch.Tensor):
                Input features for each node,
                size=[num_nodes, num_node_features]
            bond_info (torch.Tensor):
                Bond type information packed into a single matrix,
                type: torch.long, shape: [-1, 3],
                where 3 = begin_ids + end_ids + bond_type
            cond_features (torch.Tensor or None):
                Input conditional features,
                should be None if self.conditional is False

        Returns:
            torch.Tensor:
                Output feature for each node,
                size=[num_nodes, hidden_sizes[-1]]
        """
        atom_features_out = atom_features
        layer: MolConv
        for layer in self.layers:
            atom_features_out = layer(atom_features_out,
                                      bond_info,
                                      cond_features)
        return atom_features_out
