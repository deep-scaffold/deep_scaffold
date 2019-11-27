"""Operations"""
# region
import typing as t

import numpy as np
from scipy import sparse
import torch
from torch import nn
import torch_scatter

from mol_spec import MoleculeSpec
# endregion


__all__ = ['get_activation',
           'pad_first',
           'rep_and_range',
           'pack_encoder',
           'pack_decoder']


# pylint: disable=invalid-name
def segment_softmax_with_bias(x: torch.Tensor,
                              bias: torch.Tensor,
                              seg_ids: torch.Tensor,
                              eps: float = 1e-6) -> t.Tuple[torch.Tensor,
                                                            torch.Tensor]:
    """Segment softmax with bias

    Args:
        x (torch.Tensor): Input tensor, with shape [N, F]
        bias (torch.Tensor): Input bias, with shape [num_seg, ]
        seg_ids (torch.Tensor): Vector of size N
        eps (float): A small value for numerical stability

    Returns:
        tuple[torch.Tensor]
    """

    # get shape information
    num_seg = bias.size(0)

    # The max trick
    # size: [N, F + 1]
    # pylint: disable=bad-continuation
    x_max: torch.Tensor = torch.cat([x,
                                     bias.index_select(0, seg_ids)
                                         .unsqueeze(-1)],
                                    dim=-1)
    # size: [N, ]
    x_max, _ = torch.max(x_max, dim=-1)
    # size: [num_seg, ]
    x_max, _ = torch_scatter.scatter_max(x_max,
                                         index=seg_ids,
                                         dim=0,
                                         dim_size=num_seg)

    x = x - x_max.index_select(0, seg_ids).unsqueeze(-1)
    bias = bias - x_max

    x_exp, bias_exp = torch.exp(x), torch.exp(bias)
    # shape: [num_seg, ]
    x_sum = torch_scatter.scatter_add(x_exp.sum(-1), dim=0,
                                      index=seg_ids, dim_size=num_seg)
    # shape: [num_seg, ]
    x_bias_sum = x_sum + bias_exp + eps
    # shape: [N, F]
    x_softmax = x_exp / x_bias_sum.index_select(0, seg_ids).unsqueeze(-1)
    # shape: [num_seg, ]
    bias_softmax = bias_exp / x_bias_sum

    return x_softmax, bias_softmax


def get_activation(name: str,
                   *args,
                   **kwargs):
    """ Get activation module by name

    Args:
        name (str): The name of the activation function (relu, elu, selu)
        args, kwargs: Other parameters

    Returns:
        nn.Module: The activation module
    """
    name = name.lower()
    if name == 'relu':
        return nn.ReLU(*args, **kwargs)
    elif name == 'elu':
        return nn.ELU(*args, **kwargs)
    elif name == 'selu':
        return nn.SELU(*args, **kwargs)
    else:
        raise ValueError('Activation not implemented')


def pad_first(x: torch.Tensor) -> torch.Tensor:
    """ Add a single zero padding at the beginning of x

    Args:
        x (torch.Tensor):
            The input tensor.
            The dimension of x should be 1 and the type of x should be
            `torch.long`

    Returns:
        torch.Tensor: The output tensor
    """
    assert len(x.shape) == 1
    # pylint: disable=not-callable
    padding = torch.tensor([0, ], dtype=torch.long, device=x.device)
    return torch.cat([padding, x])


def rep_and_range(x: torch.Tensor):
    """
    Examples:
        (3, 2, 5) -> (0, 0, 0, 1, 1, 2, 2, 2, 2, 2) ,
                     (0, 1, 2, 0, 1, 0, 1, 2, 3, 4)

    Args:
        x (torch.Tensor): The input one-dimensional torch.long tensor
    """
    # one dimensional input
    assert len(x.shape) == 1
    total_count = x.sum()
    range_rep = torch.arange(total_count, dtype=torch.long, device=x.device)
    cum_sum = torch.cumsum(x, dim=0)
    rep_range = torch.zeros_like(range_rep)
    rep_range[cum_sum[:-1]] = 1
    rep_range = torch.cumsum(rep_range, dim=0)
    cum_sum = pad_first(cum_sum[:-1])
    range_rep = range_rep - cum_sum[rep_range]
    return rep_range, range_rep


# pylint: disable=too-many-statements
def pack_decoder(mol_array: torch.Tensor,
                 ms=MoleculeSpec.get_default()) -> t.Tuple[torch.Tensor, ...]:
    """
    Pack and expand information in mol_array in order to feed into the neural
    network

    Args:
        mol_array (torch.Tensor):
            input molecule array,
            size [batch_size, max_num_steps, 5], type: `torch.long`
            5 = atom_type + begin_ids + end_ids + bond_type + is_scaffold
        ms (mol_spec.MoleculeSpec)

    Returns:
        atom_types (torch.Tensor):
            Atom type information packed into a single vector, type: torch.long
        is_scaffold (torch.Tensor):
            Whether the corresponding atom is contained in the scaffold,
            type: torch.long
        bond_info (torch.Tensor):
            Bond type information packed into a single matrix,
            type: torch.long, shape: [-1, 3]
            3 = begin_ids + end_ids + bond_type
        actions (torch.Tensor):
        The action to carry out at each step, type: torch.long, shape: [-1, 5]
            5 = action_type + atom_type + bond_type + append_loc + connect_loc
        mol_ids, step_ids, block_ids (torch.Tensor):
            Index information, type: torch.long
    """
    # get device info
    device = mol_array.device

    # magic numbers
    I_ATOM_TYPE, I_BEGIN_IDS, I_END_IDS, I_BOND_TYPE, I_IS_SCAFFOLD = range(5)

    # The number of decoding steps required for the entire molecule
    # size: [batch_size, ]
    num_total_steps = mol_array[:, :, I_END_IDS].ge(0).long().sum(-1)
    # The number of steps required for the generation of scaffold,
    # size: [batch_size, ]
    num_scaffold_steps = mol_array[:, :, I_IS_SCAFFOLD].eq(1).long().sum(-1)
    # The number of steps required for the generation of side chains,
    # size: [batch_size, ]
    # NOTE: The additional 1 step is the termination step
    num_steps = num_total_steps - num_scaffold_steps + 1

    # Get molecule id and step id for each (unexpanded) atom/node
    # Example:
    # if we have num_steps = [4, 2, 5] and batch_size = 3, then
    # molecule:  |   0   | 1 |    2    |
    # mol_ids =  [0 0 0 0 1 1 2 2 2 2 2]
    # step_ids = [0 1 2 3 0 1 0 1 2 3 4]
    mol_ids, step_ids = rep_and_range(num_steps)

    # Expanding molecule
    # Example:
    # if we have num_steps = [3, 2],
    #            batch_size = 2,
    #            step_ids = [0 1 2 0 1]
    #            and num_scaffold_steps=[3, 2], then
    # molecule:     |           0           |    1    |
    # num_steps:    |           3           |    2    |
    # steps:        |  0  |   1   |    2    | 0 |  1  |
    # rep_ids_rep = [0 0 0 1 1 1 1 2 2 2 2 2 3 3 4 4 4]
    # indexer     = [0 1 2 0 1 2 3 0 1 2 3 4 0 1 0 1 2]
    (rep_ids_rep,
     indexer) = rep_and_range(step_ids + num_scaffold_steps[mol_ids])

    # Expanding mol_ids
    # Example:
    # molecule:     |           0           |    1    |
    # mol_ids:      |  0  |   0   |    0    | 1 |  1  |
    # rep_ids_rep = [0 0 0 1 1 1 1 2 2 2 2 2 3 3 4 4 4]
    # mol_ids_rep = [0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1]
    mol_ids_rep = mol_ids[rep_ids_rep]

    # Expanding and packing mol_array
    # Example:
    # if we have
    # mol_array = a1 a2 a3 a4 a5
    #             b1 b2 b3 -- --
    # where: a1 = [atom_type, begin_ids, end_ids, bond_type is_scaffold]
    #        -- = [-1         -1         -1       -1        -1         ]
    # then
    # indexer          = [0  1  2  0  1  2  3  0  1  2  3  4  0  1  0  1  2 ]
    # mol_ids_rep      = [0  0  0  0  0  0  0  0  0  0  0  0  1  1  1  1  1 ]
    # mol_array_packed = [a1 a2 a3 a1 a2 a3 a4 a1 a2 a3 a4 a5 b1 b2 b1 b2 b3]
    # shape: [-1, 4]
    mol_array_packed = mol_array[mol_ids_rep, indexer, :]

    # Get the expanded atom type
    atom_types, is_scaffold = (mol_array_packed[:, I_ATOM_TYPE],
                               mol_array_packed[:, I_IS_SCAFFOLD])

    # Get the number of atom at each step: |V_i|
    # Example:
    # molecule:     |           0           |    1    |
    # steps:        |  0  |   1   |    2    | 0 |  1  |
    # num_atoms =   [  3  ,   4   ,    5    , 2 ,  3  ]
    # Note: connect actions should be first filtered
    # binary vector with int values
    is_connect = atom_types.ge(0).long()
    is_connect_index = is_connect.nonzero().squeeze()
    num_atoms = torch_scatter.scatter_add(is_connect, dim=0, index=rep_ids_rep)
    atom_types = atom_types[is_connect_index]
    is_scaffold = is_scaffold[is_connect_index]

    # Get last_append_mask
    OLD_ATOM, NEW_APPEND, NEW_CONNECT = range(3)
    # Locations of the latest appended atoms
    last_append_loc = torch.cumsum(num_atoms, dim=0) - 1
    # Initialize last_append_mask as zeros
    last_append_mask = torch.full_like(is_scaffold, OLD_ATOM)
    last_append_mask[last_append_loc] = torch.where(
        is_scaffold[last_append_loc].eq(1),
        torch.full_like(last_append_loc, OLD_ATOM),
        torch.where(
            num_atoms.gt(pad_first(num_atoms[:-1])),
            torch.full_like(last_append_loc, NEW_APPEND),
            torch.full_like(last_append_loc, NEW_CONNECT)))

    # block_ids, essentially equal to the filtered rep_ids_rep
    block_ids, atom_ids = rep_and_range(num_atoms)

    # Get (packed) bond information
    # size: [-1, 3], where 3=begin_ids, end_ids, bond_type
    bond_info = mol_array_packed[:, [I_BEGIN_IDS, I_END_IDS, I_BOND_TYPE]]

    # adjust begin_ids and end_ids for each bond
    num_atoms_cumsum = pad_first(torch.cumsum(num_atoms, dim=0)[:-1])
    I_BOND_BEGIN_IDS = 0
    _filter = bond_info[:, I_BOND_BEGIN_IDS].ge(0).nonzero().squeeze()
    _shift = num_atoms_cumsum[rep_ids_rep]
    _shift = torch.stack([_shift, ] * 2 + [torch.zeros_like(_shift)], dim=1)
    bond_info = (bond_info + _shift)[_filter, :]

    # symmetrize bond_info
    bond_info = torch.cat([bond_info, bond_info[:, [1, 0, 2]]], )

    # labels for artificial bonds
    (I_BOND_REMOTE_2,
     I_BOND_REMOTE_3,
     I_BOND_ATOM_SELF) = range(ms.num_bond_types, ms.num_bond_types + 3)

    # artificial bond type: remote connection
    indices = bond_info[:, :2].cpu().numpy()
    size = atom_types.size(0)
    d_indices_2, d_indices_3 = get_remote_connection(indices, size)
    # pylint: disable=not-callable
    d_indices_2, d_indices_3 = (
        torch.tensor(d_indices_2,
                     dtype=torch.long,
                     device=device),
        torch.tensor(d_indices_3,
                     dtype=torch.long,
                     device=device)
    )
    bond_type = torch.full([d_indices_2.size(0), 1],
                           I_BOND_REMOTE_2,
                           dtype=torch.long,
                           device=device)
    bond_info_remote_2 = torch.cat([d_indices_2,
                                    bond_type],
                                   dim=-1)
    bond_type = torch.full([d_indices_3.size(0), 1],
                           I_BOND_REMOTE_3,
                           dtype=torch.long,
                           device=device)
    bond_info_remote_3 = torch.cat([d_indices_3, bond_type], dim=-1)
    bond_info = torch.cat([bond_info,
                           bond_info_remote_2,
                           bond_info_remote_3], dim=0)

    # artificial bond type: self connection
    begin_ids = end_ids = torch.arange(atom_types.size(0),
                                       dtype=torch.long,
                                       device=atom_types.device)
    bond_type = torch.full_like(end_ids, I_BOND_ATOM_SELF)
    bond_info_self = torch.stack([begin_ids, end_ids, bond_type], dim=-1)
    bond_info = torch.cat([bond_info, bond_info_self], dim=0)

    # compile action for each step, which contains the following information
    # 1. the type of action to carry out: append, connect or terminate
    # 2. the type of atom to append (0 for connect and termination actions)
    # 3. the type of bond to connect (0 for termination actions)
    # 4. the location to append (0 for connect and termination actions)
    # 5. the location to connect (0 for append and termination actions)

    # get the batch size
    batch_size = mol_array.size(0)
    padding = torch.full([batch_size, 1, 5], -1,
                         dtype=torch.long,
                         device=torch.device('cuda:0'))
    actions = torch.cat([mol_array, padding], dim=1)
    actions = actions[mol_ids, step_ids + num_scaffold_steps[mol_ids], :]

    # 1. THE TYPE OF ACTION PERFORMED AT EACH STEP
    # 0 for append, 1 for connect, 2 for termination
    I_MASK, I_APPEND, I_CONNECT, I_END = 0, 0, 1, 2

    def _full(_x):
        """A helper class to create a constant matrix
        with the same type and length as actions with a given content"""
        return torch.full([actions.size(0), ],
                          _x,
                          dtype=torch.long,
                          device=mol_array.device)

    action_type = torch.where(
        # if the atom type is defined for step i
        actions[:, I_ATOM_TYPE].ge(I_MASK),
        # the the action type is set to 'append'
        _full(I_APPEND),
        torch.where(
            # if the bond type is defined for step i
            actions[:, I_BOND_TYPE].ge(I_MASK),
            # then the action is set to 'connect'
            _full(I_CONNECT),
            # else 'terminate'
            _full(I_END)))

    # 2. THE TYPE OF ATOM ADDED AT EACH 'APPEND' STEP
    action_atom_type = torch.where(actions[:, I_ATOM_TYPE].ge(I_MASK),
                                   actions[:, I_ATOM_TYPE],
                                   _full(I_MASK))

    # 3. THE BOND TYPE AT EACH STEP
    action_bond_type = torch.where(actions[:, I_BOND_TYPE].ge(0),
                                   actions[:, I_BOND_TYPE],
                                   _full(I_MASK))

    # 4. THE LOCATION TO APPEND AT EACH STEP
    append_loc = torch.where(action_type.eq(0),
                             actions[:, I_BEGIN_IDS] + num_atoms_cumsum,
                             _full(I_MASK))

    # 5. THE LOCATION TO CONNECT AT EACH STEP
    connect_loc = torch.where(action_type.eq(1),
                              actions[:, I_END_IDS] + num_atoms_cumsum,
                              _full(I_MASK))

    # Stack everything together
    # size: [-1, 5]
    # 5 = action_type, atom_type, bond_type, append_loc, connect_loc
    actions = torch.stack([action_type,
                           action_atom_type,
                           action_bond_type,
                           append_loc,
                           connect_loc], dim=-1)

    return (
        # 1. structure information:
        # atom (node) type and bond (edge) information
        atom_types,
        is_scaffold,
        bond_info,
        last_append_mask,
        # 2. action to carry out at each step
        actions,
        # 3. indices
        mol_ids,
        step_ids,
        block_ids,
        atom_ids)


def pack_encoder(mol_array,
                 ms=MoleculeSpec.get_default()) -> t.Tuple[torch.Tensor]:
    """
    Pack and expand information in mol_array in order to feed into graph
    encoders (The encoder version of the function `pack()`)

    Args:
        mol_array (torch.Tensor):
            input molecule array
            size [batch_size, max_num_steps, 5], type: `torch.long`
            where 5 = atom_type + begin_ids + end_ids + bond_type
        ms (mol_spec.MoleculeSpec)

    Returns:
        atom_types (torch.Tensor):
            Atom type information packed into a single vector, type: torch.long
        is_scaffold (torch.Tensor):
            Whether the corresponding atom is contained in the scaffold,
            type: torch.long
        bond_info (torch.Tensor):
            Bond type information packed into a single matrix
            type: torch.long, shape: [-1, 3],
            3 = begin_ids + end_ids + bond_type
        block_ids, atom_ids (torch.Tensor):
            type: torch.long, shape: [num_total_atoms, ]
    """

    # get device info
    device = mol_array.device

    # magical numbers
    I_ATOM_TYPE, I_BEGIN_IDS, I_END_IDS, I_BOND_TYPE, I_IS_SCAFFOLD = range(5)

    # The number of decoding steps required for each input molecule
    # size: [batch_size, ]
    num_steps = mol_array[:, :, I_END_IDS].ge(0).long().sum(-1)

    # Get molecule id and step id for each (unexpanded) atom/node
    # Example:
    # if we have num_steps = [4, 2, 5] and batch_size = 3, then
    # molecule:  |   0   | 1 |    2    |
    # mol_ids =  [0 0 0 0 1 1 2 2 2 2 2]
    # step_ids = [0 1 2 3 0 1 0 1 2 3 4]
    mol_ids, step_ids = rep_and_range(num_steps)

    mol_array_packed = mol_array[mol_ids, step_ids, :]

    # Get the expanded atom type
    atom_types, is_scaffold = (mol_array_packed[:, I_ATOM_TYPE],
                               mol_array_packed[:, I_IS_SCAFFOLD])
    # binary vector with int values
    is_connect = atom_types.ge(0).long()
    is_connect_index = is_connect.nonzero().squeeze()
    num_atoms = torch_scatter.scatter_add(is_connect, dim=0, index=mol_ids)
    atom_types = atom_types[is_connect_index]
    is_scaffold = is_scaffold[is_connect_index]
    block_ids, atom_ids = rep_and_range(num_atoms)

    # Get last append mask
    # Locations of the latest appended atoms
    last_append_loc = torch.cumsum(num_atoms, dim=0) - 1
    # Initialize last_append_mask as zeros
    last_append_mask = torch.full_like(is_scaffold, 0)
    # Fill the latest appended atoms as one
    last_append_mask[last_append_loc] = 1

    # Get (packed) bond information
    # size: [-1, 3], where 3=begin_ids, end_ids, bond_type
    bond_info = mol_array_packed[:, [I_BEGIN_IDS, I_END_IDS, I_BOND_TYPE]]

    # adjust begin_ids and end_ids for each bond
    num_atoms_cumsum = pad_first(torch.cumsum(num_atoms, dim=0)[:-1])
    I_BOND_BEGIN_IDS = 0
    _filter = bond_info[:, I_BOND_BEGIN_IDS].ge(0).nonzero().squeeze()
    _shift = num_atoms_cumsum[mol_ids]
    _shift = torch.stack([_shift, ] * 2 + [torch.zeros_like(_shift)], dim=1)
    bond_info = (bond_info + _shift)[_filter, :]

    # symmetrize bond_info
    bond_info = torch.cat([bond_info, bond_info[:, [1, 0, 2]]], )

    # labels for artificial bonds and atoms
    (I_BOND_REMOTE_2,
     I_BOND_REMOTE_3,
     I_BOND_ATOM_SELF) = range(ms.num_bond_types, ms.num_bond_types + 3)

    # artificial bond type: remote connection
    indices = bond_info[:, :2].cpu().numpy()
    size = atom_types.size(0)
    d_indices_2, d_indices_3 = get_remote_connection(indices,
                                                     size)
    # pylint: disable=not-callable
    d_indices_2, d_indices_3 = (
        torch.tensor(d_indices_2,
                     dtype=torch.long,
                     device=device),
        torch.tensor(d_indices_3,
                     dtype=torch.long,
                     device=device)
    )
    bond_type = torch.full([d_indices_2.size(0), 1],
                           I_BOND_REMOTE_2,
                           dtype=torch.long,
                           device=device)
    bond_info_remote_2 = torch.cat([d_indices_2, bond_type], dim=-1)
    bond_type = torch.full([d_indices_3.size(0), 1],
                           I_BOND_REMOTE_3,
                           dtype=torch.long,
                           device=device)
    bond_info_remote_3 = torch.cat([d_indices_3, bond_type], dim=-1)
    bond_info = torch.cat([bond_info,
                           bond_info_remote_2,
                           bond_info_remote_3], dim=0)

    # artificial bond type: self connection
    begin_ids = end_ids = torch.arange(atom_types.size(0),
                                       dtype=torch.long,
                                       device=atom_types.device)
    bond_type = torch.full_like(end_ids, I_BOND_ATOM_SELF)
    bond_info_self = torch.stack([begin_ids, end_ids, bond_type], dim=-1)
    bond_info = torch.cat([bond_info, bond_info_self], dim=0)

    return (atom_types,
            is_scaffold,
            bond_info,
            last_append_mask,
            block_ids,
            atom_ids)


def get_remote_connection(indices: np.ndarray,
                          size: int) -> t.Tuple[np.ndarray, ...]:
    """
    Get the remote connections in graph

    Args:
        indices (np.ndarray): The indices of the sparse matrix, size: [-1, 2]
        size (int): The size of the sparse matrix

    Returns:
        d_indices_2, d_indices_3 (np.ndarray): Remote connections
    """

    row, col, data = (indices[:, 0],
                      indices[:, 1],
                      np.ones([indices.shape[0], ], dtype=np.float32))
    adj = sparse.coo_matrix((data, (row, col)), shape=(size, size))

    d = adj * adj

    d_indices_2 = np.stack(d.nonzero(), axis=1)
    # remove diagonal elements
    d_indices_2 = d_indices_2[d_indices_2[:, 0] != d_indices_2[:, 1], :]

    d = d * adj
    d = d - d.multiply(adj)

    d_indices_3 = np.stack(d.nonzero(), axis=1)
    # remove diagonal elements
    d_indices_3 = d_indices_3[d_indices_3[:, 0] != d_indices_3[:, 1], :]

    return d_indices_2, d_indices_3
