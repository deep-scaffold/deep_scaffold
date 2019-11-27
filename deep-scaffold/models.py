"""Models"""
# region
import typing as t

import numpy as np
import torch
from torch import nn

from mol_spec import MoleculeSpec
from deep_scaffold.layers import BNReLULinear
from deep_scaffold.layers import DenseNet
from deep_scaffold.layers import AvgPooling
from deep_scaffold import ops
# endregion


__all__ = ['DeepScaffold']


class DeepScaffold(nn.Module):
    """
    Auto-regressive molecule generative model for scaffold-based
    drug discovery
    """
    def __init__(self,
                 num_atom_embedding: int,
                 causal_hidden_sizes: int,
                 num_bn_features: int,
                 num_k_features: int,
                 num_layers: int,
                 num_output_features: int,
                 efficient: bool = False,
                 ms: MoleculeSpec = MoleculeSpec.get_default(),
                 activation: str = 'elu',
                 conditional: bool = False,
                 num_cond_features: t.Optional[int] = None,
                 activation_cond: t.Optional[str] = None):
        """
        The constructor

        Args:
            num_atom_embedding (int):
                The size of the initial node embedding
            causal_hidden_sizes (tuple[int]):
                The size of hidden layers in causal weave blocks
            num_bn_features (int):
                The number of features used in bottleneck layers in each dense
                layer
            num_k_features (int):
                The growth rate of dense net
            num_layers (int):
                The number of densenet layers
            num_output_features (int):
                The number of output features for the densenet
            efficient (bool):
                Whether to use the memory efficient BNReLULinearimplementation
                of densenet
            ms (mol_spec.MoleculeSpec)
            activation (str):
                The activation function used, default to 'elu'
            conditional (bool):
                Whether to include conditional input, default to False
            num_cond_features (int or None):
                The size of conditional input, should be None if
                self.conditional is False
            activation_cond (str or None):
                Activation function used for conditional input
                should be None if self.conditional is False
        """
        super(DeepScaffold, self).__init__()

        self.num_atom_embedding = num_atom_embedding
        self.causal_hidden_sizes = causal_hidden_sizes
        self.num_bn_features = num_bn_features
        self.num_k_features = num_k_features
        self.num_layers = num_layers
        self.num_output_features = num_output_features
        self.efficient = efficient
        # pylint: disable=invalid-name
        self.ms = ms
        # 3 = 2 * remote connection + self connection
        self._num_bond_types = self.ms.num_bond_types + 3
        self._num_atom_types = self.ms.num_atom_types
        self.activation = activation
        self.conditional = conditional
        self.num_cond_features = num_cond_features
        self.activation_cond = activation_cond

        # embedding layer for atom types and bond types
        # 3 = is_scaffold + new_append + new_connect
        self.atom_embedding = nn.Embedding((self._num_atom_types +
                                            self.ms.num_atom_types * 3),
                                           self.num_atom_embedding)

        # convolution layer
        self.mol_conv = DenseNet(self.num_atom_embedding,
                                 self._num_bond_types,
                                 self.causal_hidden_sizes,
                                 self.num_bn_features,
                                 self.num_k_features,
                                 self.num_layers,
                                 self.num_output_features,
                                 self.efficient,
                                 self.activation,
                                 self.conditional,
                                 self.num_cond_features,
                                 self.activation_cond)

        # Pooling layer
        self.avg_pool = AvgPooling(self.num_output_features,
                                   self.activation)

        # output layers
        self.end = BNReLULinear(self.num_output_features,
                                1,
                                self.activation)
        self.append_connect = \
            BNReLULinear(self.num_output_features * 2,
                         ms.num_atom_types * ms.num_bond_types +
                         ms.num_bond_types,
                         self.activation)

    def _forward(self,
                 atom_types: torch.Tensor,
                 is_scaffold: torch.Tensor,
                 bond_info: torch.Tensor,
                 block_ids: torch.Tensor,
                 last_append_mask: torch.Tensor,
                 cond_features: t.Optional[torch.Tensor] = None,
                 cond_ids: t.Optional[torch.Tensor] = None,
                 log_prob: bool = False) -> t.Tuple[torch.Tensor, ...]:
        """
        Perform the forward pass of the network

        Args:
            atom_types (torch.Tensor):
                Atom type information packed into a single vector,
                type: torch.long
            is_scaffold (torch.Tensor):
                The location information of each scaffold within the molecule,
                type: torch.long
            bond_info (torch.Tensor):
                Bond type information packed into a single matrix
                type: torch.long, shape: [-1, 3],
                where 3 = begin_ids + end_ids + bond_type
            block_ids (torch.Tensor):
                type: torch.long
            last_append_mask (torch.Tensor):
                A label used to identify which atom is latest appended
            cond_features (torch.Tensor or None):
                Input conditional features
                type: torch.float32, shape: [num_blocks, num_cond_features]
                should be None if self.conditional is False
            cond_ids (torch.Tensor or None):
                Label the conditional feature each atom corresponds to
            log_prob (bool):
                Whether to output log probability, default to False

        Returns:
            p_append (torch.Tensor):
                p_append[i, j, k] - The probability of appending atom k to
                location i with bond type j
                type: torch.float, shape: [-1, num_atom_types, num_bond_types ]
            p_connect (torch.Tensor):
                p_connect[i, j]: The probability of connecting the latest
                appended atom to atom i with bond j
                type: torch.float, shape: [-1, num_bond_types]
            p_end (torch.Tensor):
                The probability to terminate the generation,
                type: torch.float32, shape [-1]

        """
        # # get the total number of blocks
        # num_blocks = block_ids.max() + 1

        # Differentiate scaffold nodes and side-chain nodes
        atom_types = torch.where(
            is_scaffold.eq(1),
            # If the atom is inside the scaffold
            atom_types + self._num_atom_types * 1,
            # If the atom is an ordinary atom
            torch.where(
                last_append_mask.eq(1),
                # If the atom is latest appended
                atom_types + self._num_atom_types * 2,
                torch.where(
                    last_append_mask.eq(2),
                    atom_types + self._num_atom_types * 3,
                    # If the atom is not latest appended
                    atom_types)))

        atom_types = torch.where(is_scaffold.eq(1),
                                 atom_types + self._num_atom_types,
                                 atom_types)

        # get input features for nodes
        # size: [n_total_atoms, num_node_embeddings]
        atom_features = self.atom_embedding(atom_types)

        # get shape and device info
        num_atoms = atom_features.size(0)
        device = atom_features.device

        # split bond_info
        (begin_ids,
         end_ids,
         bond_types) = (bond_info[:, 0],
                        bond_info[:, 1],
                        bond_info[:, 2])

        # build adj matrix
        end_ids = end_ids + bond_types * num_atoms
        if self.conditional:
            assert cond_features is not None
            assert cond_ids is not None
            num_cond = cond_features.size(0)
            end_ids_cond = cond_ids + self._num_bond_types * num_atoms
            begin_ids_cond = torch.arange(num_atoms,
                                          dtype=torch.long,
                                          device=device)
            end_ids = torch.cat([end_ids, end_ids_cond])
            begin_ids = torch.cat([begin_ids, begin_ids_cond])
            size = torch.Size([num_atoms,
                               num_atoms * self._num_bond_types + num_cond])
        else:
            assert cond_features is None
            assert cond_ids is None
            size = torch.Size([num_atoms,
                               num_atoms * self._num_bond_types])
        indices = torch.stack([begin_ids, end_ids], dim=0)
        val: torch.Tensor = torch.ones([indices.size(-1), ],
                                       dtype=torch.float32,
                                       device=device)

        if val.is_cuda:
            adj = torch.cuda.sparse.FloatTensor(indices, val, size)
        else:
            adj = torch.sparse.FloatTensor(indices, val, size)

        bond_info = adj

        # feed into the network
        # size: [n_total_atoms, num_rev_features]
        output_features = self.mol_conv(atom_features,
                                        bond_info,
                                        cond_features)
        # For convolution with no master node
        atom_features = output_features
        mol_features = self.avg_pool(output_features,
                                     block_ids)

        # get the activation value for each action
        # size: [n_total_atoms,
        #        num_atom_types * num_bond_types + num_bond_types]
        atom_features = torch.cat([atom_features,
                                   mol_features[block_ids, :]],
                                  dim=-1)
        activation_append_connect = self.append_connect(atom_features)
        # size: [n_total_steps, 1]
        activation_end = self.end(mol_features)
        # size: [n_total_steps, ]
        activation_end = activation_end.squeeze(-1)

        # normalize to get the probability value for each action
        # shape unchanged
        (p_append_connect,
         p_end) = ops.segment_softmax_with_bias(activation_append_connect,
                                                activation_end,
                                                block_ids)

        # break append and connect
        p_append, p_connect = torch.split(p_append_connect,
                                          [(self.ms.num_atom_types *
                                            self.ms.num_bond_types),
                                           self.ms.num_bond_types],
                                          dim=-1)
        # shape: [n_total_atoms, num_atom_types, num_bond_types]
        p_append = p_append.view(-1,
                                 self.ms.num_atom_types,
                                 self.ms.num_bond_types)

        if log_prob:
            # get log likelihood
            # shape unchanged
            return (torch.log(p_append + 1e-6),
                    torch.log(p_connect + 1e-6),
                    torch.log(p_end + 1e-6))
        return p_append, p_connect, p_end

    def _generate_step(self,
                       mol_array: torch.Tensor,
                       last_action_type: t.Optional[torch.Tensor] = None,
                       cond_features: t.Optional[torch.Tensor] = None,
                       is_init: bool = False) -> t.Tuple[torch.Tensor,
                                                         torch.Tensor]:
        """
        Perform one molecule generation step

        Args:
            mol_array (torch.Tensor):
                The tensor containing the part of structural information of
                molecules that are already generated
                size [num_samples, num_steps, 4], type: `torch.long`
                5 = atom_type + begin_ids + end_ids + bond_type + is_scaffold
                NOTE: all molecules should have the same amount of steps
            last_action_type (torch.Tensor or None):
                The type of the last action, 0 for append, 1 for connect
                size: [num_samples]
            cond_features (torch.Tensor or None):
                Input conditional features, type: `torch.float32`,
                shape: [num_samples, num_cond_features]
                should be None if self.conditional is False
            is_init (bool):
                A boolean value indicating whether this step is initialization
                step

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                `mol_array_new` - The updated molecule tensor,
                size: [num_samples, num_steps + 1, 4]
                `new_action_type` - The types of actions carried out,
                -1 for terminate, 0 for append, 1 for connect

        """

        # get device and shape information
        device = mol_array.device

        # get the location of latest appended atom
        # pylint: disable=invalid-name
        I_END = 2
        # size: [num_samples, ]
        loc_latest_append, _ = mol_array[:, :, I_END].max(dim=-1)

        (atom_types,
         is_scaffold,
         bond_info,
         last_append_mask,
         block_ids,
         atom_ids) = ops.pack_encoder(mol_array,
                                      self.ms)
        if is_init:
            last_append_mask = torch.zeros_like(last_append_mask)
        else:
            last_action_type = last_action_type[block_ids]
            # 0=normal, 1=append, 2=connect
            last_append_mask = last_append_mask * (1 + last_action_type)

        def _len(_x):
            return np.asscalar(_x.max().detach().cpu().numpy()) + 1

        num_blocks, max_num_atoms = _len(block_ids), _len(atom_ids)

        if self.conditional:
            assert cond_features is not None
            assert cond_features.size(0) == mol_array.size(0)
            cond_ids = block_ids
        else:
            assert cond_features is None
            cond_ids = None

        # Get the probability for each type of actions
        # shapes:
        # p_append: [-1, num_atom_types, num_bond_types]
        # p_connect: [-1, num_bond_types]
        # p_end: [-1]
        p_append, p_connect, p_end = self._forward(atom_types,
                                                   is_scaffold,
                                                   bond_info,
                                                   block_ids,
                                                   last_append_mask,
                                                   cond_features,
                                                   cond_ids,
                                                   log_prob=False)

        # concatenate append, connect and end actions into a single matrix
        # shape: [-1, num_atom_types * num_bond_types + num_bond_types]
        p_append = p_append.view(-1,
                                 (self.ms.num_atom_types *
                                  self.ms.num_bond_types))
        p_append_connect = torch.cat([p_append, p_connect], dim=-1)

        p_append_connect_unpack = torch.zeros([num_blocks, max_num_atoms,
                                               self.ms.num_atom_types *
                                               self.ms.num_bond_types +
                                               self.ms.num_bond_types],
                                              dtype=torch.float32,
                                              device=device)
        p_append_connect_unpack[block_ids, atom_ids, :] = p_append_connect

        p_append_connect_unpack = p_append_connect_unpack.view(num_blocks, -1)
        # shape: [num_samples, action_space_size]
        # pylint: disable=invalid-name
        p = torch.cat([p_append_connect_unpack, p_end.unsqueeze(-1)], dim=-1)

        # sample action
        action = torch.multinomial(p, 1).squeeze(-1)  # shape: [num_samples, ]

        # If the new action is append
        bond_type = torch.remainder(action, self.ms.num_bond_types)
        quotient = (action - bond_type) / self.ms.num_bond_types
        atom_types = torch.remainder(quotient, self.ms.num_atom_types + 1)
        quotient = (quotient - atom_types) / (self.ms.num_atom_types + 1)
        begin_ids = torch.remainder(quotient, max_num_atoms)
        end_ids = loc_latest_append + 1
        mol_array_append = torch.stack([atom_types,
                                        begin_ids,
                                        end_ids,
                                        bond_type,
                                        torch.zeros_like(atom_types)], dim=-1)

        # if the new action is connect
        end_ids = begin_ids
        begin_ids = loc_latest_append
        mol_array_connect = torch.stack([torch.full_like(atom_types, -1),
                                         begin_ids, end_ids, bond_type,
                                         torch.zeros_like(atom_types)], dim=-1)

        # if the new action is terminate
        mol_array_end = torch.full_like(mol_array_append, -1)

        # construct the new mol_array
        is_terminate = action.eq(p.size(-1) - 1).unsqueeze(-1).expand(-1, 5)
        is_connect = atom_types.eq(self.ms.num_atom_types)\
                               .unsqueeze(-1)\
                               .expand(-1, 5)
        mol_array_new = torch.where(
            # if the action is to terminate
            is_terminate,
            mol_array_end,
            torch.where(
                # elif the action is to connect
                is_connect,
                mol_array_connect,
                # else
                mol_array_append))

        # Get the new action type
        is_terminate = is_terminate[:, 0]
        is_connect = is_connect[:, 0]
        new_action_type = torch.where(
            is_terminate,
            torch.full_like(is_terminate.long(), -1),
            torch.where(
                is_connect,
                torch.full_like(is_connect.long(), 1),
                torch.full_like(is_connect.long(), 0)))

        # update mol_array
        # size: num_samples, -1, 5
        row = torch.arange(num_blocks,
                           dtype=torch.long,
                           device=mol_array.device)
        col = mol_array[:, :, I_END].ge(0).long().sum(-1)
        mol_array[row, col, :] = mol_array_new  # Update in-place

        return mol_array, new_action_type

    @torch.no_grad()
    def generate(self,
                 scaffold_array: torch.Tensor,
                 cond_features: t.Optional[torch.Tensor] = None
                 ) -> torch.Tensor:
        """
        Generate molecules

        Args:
            scaffold_array (torch.Tensor):
                The structure of molecular scaffolds
            cond_features (torch.Tensor or None):
                Input conditional features,
                type: torch.float32,
                shape: [num_samples, num_cond_features] or [num_cond_features,]
                should be None if self.conditional is False

        Returns:
            torch.Tensor:
                `mol_array` - The generated molecules
        """
        num_samples, max_num_scaffold_steps, _ = scaffold_array.shape
        device = scaffold_array.device

        if self.conditional:
            assert cond_features is not None
            if len(cond_features.shape) == 1:
                cond_features = cond_features.unsqueeze(0)\
                                             .expand(num_samples, -1)
            else:
                assert cond_features.size(0) == num_samples
        else:
            assert cond_features is None

        # step 1: initialization
        padding_size = self.ms.max_num_bonds - max_num_scaffold_steps + 1
        padding = torch.full(size=[num_samples,
                                   padding_size,
                                   5],
                             fill_value=-1,
                             dtype=torch.long,
                             device=device)
        # size: num_samples, max_num_bonds + 1, 5
        mol_array = torch.cat([scaffold_array, padding], dim=1)
        # pylint: disable=unsubscriptable-object
        last_action_type = None
        is_terminated = torch.Tensor(num_samples)\
                             .zero_()\
                             .byte()\
                             .to(scaffold_array.device)

        # step 2: extend
        for step_id in range(padding_size):
            # if generation have completed for all molecules
            if is_terminated.all():
                # break the loop
                break
            # else

            # Record which molecule have not finished generation
            not_terminated = ~is_terminated  # type: torch.Tensor
            not_terminated_index = not_terminated.nonzero().squeeze(-1)

            # Extract non-finished molecules
            mol_array_non_finished = mol_array[not_terminated_index, :, :]
            cond_features_non_finished = \
                (cond_features[not_terminated_index, :]
                 if cond_features is not None
                 else None)
            if last_action_type is None:
                last_action_type_non_finished = None
            else:
                last_action_type_non_finished = \
                    last_action_type[not_terminated_index]

            # expand one step
            new_action_type: torch.Tensor
            (mol_array_new,
             new_action_type) = \
                self._generate_step(mol_array_non_finished,
                                    last_action_type_non_finished,
                                    cond_features_non_finished,
                                    is_init=(step_id == 0))

            # update mol_array
            mol_array[not_terminated_index, :, :] = mol_array_new
            if last_action_type is None:
                last_action_type = new_action_type
            else:
                last_action_type[not_terminated_index] = new_action_type
            is_terminated = last_action_type.eq(-1)

        return mol_array

    def likelihood(self,
                   mol_array: torch.Tensor,
                   cond_features: t.Optional[torch.Tensor] = None
                   ) -> torch.Tensor:
        """
        Calculate the likelihood value for the input molecules

        Args:
            mol_array (torch.Tensor):
                The tensor containing the structural information of input
                molecules
                size [batch_size, max_num_steps, 5], type: `torch.long`
                5 = atom_type + begin_ids + end_ids + bond_type + is_scaffold
            cond_features (torch.Tensor or None):
                Input conditional features
                type: torch.float32, shape: [batch_size, num_cond_features]
                should be None if self.conditional is False

        Returns:
            ll_unpacked (torch.Tensor):
                The likelihood value for each step
                size [batch_size, max_num_steps + 1], type: `torch.float32`
                value ll_unpacked[i, j] indicates the log-likelihood value for
                molecule i at step j
        """

        # get device and shape information
        device = mol_array.device
        batch_size, max_num_steps, _ = mol_array.shape

        with torch.no_grad():
            # pack up molecules
            [atom_types,  # size: [n_total_atoms, ]
             is_scaffold,  # size: [n_total_atoms, ]
             bond_info,  # size: [n_total_bonds, 3],
                         # where 3=begin_ids+end_ids+bond_type
             last_append_mask,  # size: [n_total_atoms,]
             actions,  # size: [n_total_steps, 5],
                       # where 5=action_types + atom_type + bond_type +
                       #         append_pos + connect_pos
             mol_ids, step_ids,  # size: [n_total_steps, ]
             block_ids,  # size: [n_total_atoms, ]
             _] = ops.pack_decoder(mol_array, self.ms)

        # prepare conditional features and log_p_0
        if self.conditional:
            assert cond_features is not None
            assert cond_features.size(0) == mol_array.size(0)
            cond_ids = mol_ids[block_ids]
        else:
            assert cond_features is None
            cond_ids = None

        # get log likelihood for append, connect and termination actions
        (log_p_append,
         log_p_connect,
         log_p_end) = self._forward(atom_types,
                                    is_scaffold,
                                    bond_info,
                                    block_ids,
                                    last_append_mask,
                                    cond_features,
                                    cond_ids,
                                    log_prob=True)

        # calculate likelihood

        # decompose action
        # shape: [n_total_steps, ]
        (action_types,
         atom_types_t,
         bond_types_t,
         append_pos,
         connect_pos) = \
            (actions[:, 0],
             actions[:, 1],
             actions[:, 2],
             actions[:, 3],
             actions[:, 4])

        # pylint: disable=invalid-name
        I_APPEND, I_CONNECT, I_END = 0, 1, 2

        # get likelihood for termination actions
        # size: [n_total_steps, ]
        ll_end = torch.where(action_types.eq(I_END),
                             log_p_end,
                             torch.zeros_like(log_p_end))

        # get likelihood for append actions
        (_indices_1,
         _indices_2,
         _indices_3) = (append_pos,
                        atom_types_t,
                        bond_types_t)
        ll_append = log_p_append[_indices_1, _indices_2, _indices_3]
        # size: [n_total_steps, ]
        ll_append = torch.where(action_types.eq(I_APPEND),
                                ll_append,
                                torch.zeros_like(ll_append))

        # get likelihood for connect actions
        _row, _col = connect_pos, bond_types_t
        ll_connect = log_p_connect[_row, _col]
        # size: [n_total_steps, ]
        ll_connect = torch.where(action_types.eq(I_CONNECT),
                                 ll_connect,
                                 torch.zeros_like(ll_connect))

        # total
        # size: [n_total_steps, ]
        # pylint: disable=invalid-name
        ll = ll_append + ll_connect + ll_end

        # organize into matrix
        ll_unpack = torch.zeros([batch_size, max_num_steps],
                                dtype=torch.float32,
                                device=device)
        # shape: [batch_size, max_num_steps + 1]
        ll_unpack[mol_ids, step_ids] = ll

        return ll_unpack

    def forward(self, *args, **kwargs):
        """
        Forward function for module
        """
        return self.likelihood(*args, **kwargs)
