"""Utility functions"""
# region
import typing as t

import networkx as nx
import numpy as np
from rdkit import Chem

from mol_spec import MoleculeSpec
# endregion


__all__ = ['get_mol_from_array', 'get_array_from_mol']


class GetOutOfLoop(Exception):
    """
    Utility exception to break multiple loops
    Adopted from:
    https://stackoverflow.com/questions/189645/\
    how-to-break-out-of-multiple-loops-in-python
    """


# SECTION Conversion between `Chem.Mol` and `np.ndarray`
# pylint: disable=invalid-name
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
def get_mol_from_array(mol_array: np.ndarray,
                       sanitize: bool = True,
                       ms: MoleculeSpec = MoleculeSpec.get_default()
                       ) -> t.List[Chem.Mol]:
    """
    Converting molecule array to Chem.Mol objects

    Args:
        mol_array (np.ndarray):
            The array representation of molecules
            dtype: int, shape: [num_samples, num_steps, 5]
        sanitize (bool):
            Whether to sanitize the output molecule, default to True
        ms (mol_spec.MoleculeSpec)

    Returns:
        list[Chem.Mol]:
            mol_list - The list of output molecules
    """
    # shape: num_samples, num_steps
    is_scaffold = mol_array[:, :, -1]
    # shape: num_samples, num_steps, 5
    mol_array = mol_array[:, :, :-1]

    # get shape information
    num_samples, max_num_steps, _ = mol_array.shape

    # initialize the list of output molecules
    mol_list = []

    # loop over molecules
    # pylint: disable=too-many-nested-blocks
    for mol_id in range(num_samples):
        try:
            mol = Chem.RWMol(Chem.Mol())  # initialize molecule
            atom_list = []  # List to store all created atoms
            scaffold_atoms = []  # List to store all scaffold atoms
            aromatic_atoms = []  # List to store all aromatic atoms
            n_atoms = []  # The indices of all nitrogen atoms
            for step_id in range(max_num_steps):
                (atom_type,
                 begin_ids,
                 end_ids,
                 bond_type) = mol_array[mol_id, step_id, :].tolist()
                if end_ids == -1:
                    # if the actions is to terminate
                    break
                elif begin_ids == -1:
                    # if the action is to initialize
                    new_atom = ms.index_to_atom(atom_type)
                    mol.AddAtom(new_atom)
                    atom_list.append(new_atom)
                    if new_atom.GetSymbol() == 'N':
                        n_atoms.append(end_ids)
                elif atom_type == -1:
                    # if the action is to connect
                    ms.index_to_bond(mol, begin_ids, end_ids, bond_type)
                else:
                    # if the action is to append new atom
                    new_atom = ms.index_to_atom(atom_type)
                    mol.AddAtom(new_atom)
                    ms.index_to_bond(mol, begin_ids, end_ids, bond_type)
                    # Record atom
                    atom_list.append(new_atom)
                    if is_scaffold[mol_id, step_id]:
                        # Both ends are scaffold atoms
                        scaffold_atoms.append(end_ids)
                        scaffold_atoms.append(begin_ids)
                    if bond_type == \
                            ms.bond_orders.index(Chem.BondType.AROMATIC):
                        aromatic_atoms.append(begin_ids)
                        aromatic_atoms.append(end_ids)
                    if new_atom.GetSymbol() == 'N':
                        n_atoms.append(end_ids)

            special_atoms = (set(scaffold_atoms) &
                             set(aromatic_atoms) &
                             set(n_atoms))
            scaffold_atoms = set(scaffold_atoms)

            for atom_id in special_atoms:
                neighbors = (mol_array[mol_id,
                                       mol_array[mol_id, :, 2] == atom_id,
                                       1].tolist() +
                             mol_array[mol_id,
                                       mol_array[mol_id, :, 1] == atom_id,
                                       2].tolist())
                neighbors = set(neighbors) - {-1}
                if neighbors - scaffold_atoms:
                    atom_i = mol.GetAtomWithIdx(atom_id)
                    if atom_i.GetNumExplicitHs() > 0:
                        num_explict_hs = atom_i.GetNumExplicitHs() - 1
                        atom_i.SetNumExplicitHs(num_explict_hs)
                    else:
                        num_formal_charge = atom_i.GetFormalCharge() + 1
                        atom_i.SetFormalCharge(num_formal_charge)

            if sanitize:
                mol = mol.GetMol()
                Chem.SanitizeMol(mol)
        except (ValueError, RuntimeError):
            mol = None
        mol_list.append(mol)

    return mol_list


def get_array_from_mol(mol: Chem.Mol,
                       scaffold_nodes: t.Iterable,
                       nh_nodes: t.Iterable,
                       np_nodes: t.Iterable,
                       k: int,
                       p: float,
                       ms: MoleculeSpec = MoleculeSpec.get_default()
                       ) -> t.Tuple[np.ndarray, np.ndarray]:
    """
    Represent the molecule using `np.ndarray`

    Args:
        mol (Chem.Mol):
            The input molecule
        scaffold_nodes (Iterable):
            The location of scaffold represented as `list`/`np.ndarray`
        nh_nodes (Iterable):
            Nodes with modifications
        np_nodes (Iterable):
            Nodes with modifications
        k (int):
            The number of importance samples
        p (float):
            Degree of uncertainty during route sampling, should be in (0, 1)
        ms (mol_spec.MoleculeSpec)

    Returns:
        mol_array (np.ndarray):
            The numpy representation of the molecule
            dtype - np.int32, shape - [k, num_bonds + 1, 5]
        logp (np.ndarray):
            The log-likelihood of each route
            dtype - np.float32, shape - [k, ]
    """
    atom_types, bond_info = [], []
    _, num_bonds = mol.GetNumAtoms(), mol.GetNumBonds()

    # sample route
    scaffold_nodes = np.array(list(scaffold_nodes), dtype=np.int32)
    route_list, step_ids_list, logp = _sample_ordering(mol,
                                                       scaffold_nodes,
                                                       k,
                                                       p)

    for atom_id, atom in enumerate(mol.GetAtoms()):
        if atom_id in nh_nodes:
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
        if atom_id in np_nodes:
            atom.SetFormalCharge(atom.GetFormalCharge() - 1)
        atom_types.append(ms.get_atom_type(atom))

    for bond in mol.GetBonds():
        bond_info.append([bond.GetBeginAtomIdx(),
                          bond.GetEndAtomIdx(),
                          ms.get_bond_type(bond)])

    # shape:
    # atom_types: num_atoms
    # bond_info: num_bonds x 3
    atom_types, bond_info = (np.array(atom_types, dtype=np.int32),
                             np.array(bond_info, dtype=np.int32))

    # initialize packed molecule array data
    mol_array = []

    for sample_id in range(k):
        # get the route and step_ids for the i-th sample
        (route_i,
         step_ids_i) = (route_list[sample_id, :],
                        step_ids_list[sample_id, :])

        # reorder atom types and bond info
        # note: bond_info [start_ids, end_ids, bond_type]
        (atom_types_i,
         bond_info_i,
         is_append) = _reorder(atom_types,
                               bond_info,
                               route_i,
                               step_ids_i)

        # atom type added at each step
        # -1 if the current step is connect
        atom_types_added = np.full([num_bonds, ],
                                   -1,
                                   dtype=np.int32)
        atom_types_added[is_append] = \
            atom_types_i[bond_info_i[:, 1]][is_append]

        # pack into mol_array_i
        # size: num_bonds x 4
        # note: [atom_types_added, start_ids, end_ids, bond_type]
        mol_array_i = np.concatenate([atom_types_added[:, np.newaxis],
                                      bond_info_i],
                                     axis=-1)

        # add initialization step
        init_step = np.array([[atom_types_i[0], -1, 0, -1]], dtype=np.int32)

        # concat into mol_array
        # size: (num_bonds + 1) x 4
        mol_array_i = np.concatenate([init_step, mol_array_i], axis=0)

        # Mark up scaffold bonds
        is_scaffold = np.logical_and(mol_array_i[:, 1] < len(scaffold_nodes),
                                     mol_array_i[:, 2] < len(scaffold_nodes))
        is_scaffold = is_scaffold.astype(np.int32)

        # Concatenate
        # shape: k x (num_bonds + 1) x 5
        mol_array_i = np.concatenate((mol_array_i,
                                      is_scaffold[:, np.newaxis]),
                                     axis=-1)

        mol_array.append(mol_array_i)

    # num_samples x (num_bonds + 1) x 4
    mol_array = np.stack(mol_array, axis=0)

    # Output size:
    # mol_array: k x (num_bonds + 1) x 4
    # logp: k

    return mol_array, logp
# !SECTION


# SECTION Helper functions
def _sample_ordering(mol: Chem.Mol,
                     scaffold_nodes: np.ndarray,
                     k: int,
                     p: float,
                     ms: MoleculeSpec = MoleculeSpec.get_default()
                     ) -> t.Tuple[np.ndarray,
                                  np.ndarray,
                                  np.ndarray]:
    """Sampling decoding routes of a given molecule `mol`

    Args:
        mol (Chem.Mol):
            the given molecule (type: Chem.Mol)
        scaffold_nodes (np.ndarray):
            the nodes marked as scaffold
        k (int):
            The number of importance samples
        p (float):
            Degree of uncertainty during route sampling, should be in (0, 1)
        ms (mol_spec.MoleculeSpec)

    Returns:
        route_list (np.ndarray):
            route_list[i][j]
            the index of the atom reached at step j in sample i
        step_ids_list (np.ndarray):
            step_ids_list[i][j]
            the step at which atom j is reach at sample i
        logp_list (np.ndarray):
            logp_list[i] - the log-likelihood value of route i
    """
    # build graph
    atom_types, atom_ranks, bonds = [], [], []
    for atom in mol.GetAtoms():
        atom_types.append(ms.get_atom_type(atom))
    for r in Chem.CanonicalRankAtoms(mol):
        atom_ranks.append(r)
    for b in mol.GetBonds():
        idx_1, idx_2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bonds.append([idx_1, idx_2])
    atom_ranks = np.array(atom_ranks)

    # build nx graph
    graph = nx.Graph()
    graph.add_nodes_from(range(len(atom_ranks)))
    graph.add_edges_from(bonds)

    route_list, step_ids_list, logp_list = [], [], []
    for _ in range(k):
        step_ids, log_p = _traverse(graph, atom_ranks, scaffold_nodes, p)
        step_ids_list.append(step_ids)
        step_ids = np.argsort(step_ids)
        route_list.append(step_ids)
        logp_list.append(log_p)

    # cast to numpy array
    (route_list,
     step_ids_list) = (np.array(route_list, dtype=np.int32),
                       np.array(step_ids_list, dtype=np.int32))
    logp_list = np.array(logp_list, dtype=np.float32)

    return route_list, step_ids_list, logp_list


def _reorder(atom_types: np.ndarray,
             bond_info: np.ndarray,
             route: np.ndarray,
             step_ids: np.ndarray
             )-> t.Tuple[np.ndarray,
                         np.ndarray,
                         np.ndarray]:
    """ Reorder atom and bonds according the decoding route

    Args:
        atom_types (np.ndarray):
            storing the atom type of each atom, size: num_atoms
        bond_info (np.ndarray):
            storing the bond information, size: num_bonds x 3
        route (np.ndarray):
            route index
        step_ids (np.ndarray):
            step index

    Returns:
        atom_types, bond_info, is_append (np.ndarray):
            reordered atom_types and bond_info
    """

    atom_types, bond_info = np.copy(atom_types), np.copy(bond_info)

    # sort by step_ids
    atom_types = atom_types[route]
    (bond_info[:, 0],
     bond_info[:, 1]) = (step_ids[bond_info[:, 0]],
                         step_ids[bond_info[:, 1]])
    max_b, min_b = (np.amax(bond_info[:, :2], axis=1),
                    np.amin(bond_info[:, :2], axis=1))
    bond_info = bond_info[np.lexsort([-min_b, max_b]), :]

    # separate append and connect
    max_b, min_b = (np.amax(bond_info[:, :2], axis=1),
                    np.amin(bond_info[:, :2], axis=1))
    is_append = np.concatenate([np.array([True]), max_b[1:] > max_b[:-1]])
    bond_info = np.concatenate([np.where(is_append[:, np.newaxis],
                                         np.stack([min_b, max_b], axis=1),
                                         np.stack([max_b, min_b], axis=1)),
                                bond_info[:, -1:]], axis=1)

    return atom_types, bond_info, is_append


def _traverse(graph: nx.Graph,
              atom_ranks: np.ndarray,
              scaffold_nodes: np.ndarray,
              p: float):
    """ An recursive function for stochastic traversal of the entire `graph`

    Args:
        graph (nx.Graph):
            The graph to be traversed
        atom_ranks (np.ndarray):
            A list storing the rank of each atom
        scaffold_nodes (np.ndarray):
            A list storing the index of atoms in the scaffold
        p (float):
            Degree of uncertainty during route sampling, should be in (0, 1)

    Returns:
        step_ids (np.ndarray):
            `step_ids` for the next traversal step
        log_p (np.ndarray):
            `log_p` for the next traversal step
    """
    step_ids = _traverse_scaffold(graph,
                                  atom_ranks,
                                  scaffold_nodes,
                                  p)
    if len(scaffold_nodes) < len(atom_ranks):
        step_ids, log_p = _traverse_chain(graph,
                                          atom_ranks,
                                          scaffold_nodes,
                                          step_ids,
                                          p)
    else:
        log_p = 0.0

    return step_ids, log_p


def _traverse_scaffold(graph: nx.Graph,
                       atom_ranks: np.ndarray,
                       scaffold_nodes: np.ndarray,
                       p: float,
                       current_node: t.Optional[int] = None,
                       step_ids: t.Optional[np.ndarray] = None):
    """
    An recursive function for stochastic traversal of scaffold in `graph`
    """
    # Initialize next_nodes and step_ids (if is None)
    if current_node is None:
        # Initialize as the set of all scaffold nodes
        next_nodes = scaffold_nodes
        # Initialize step_ids as -1
        step_ids = np.full_like(atom_ranks, -1)
    else:
        # get neighbor nodes
        next_nodes = np.array(list(graph.neighbors(current_node)),
                              dtype=np.int32)
        # Only scaffold nodes
        next_nodes = np.intersect1d(next_nodes,
                                    scaffold_nodes,
                                    assume_unique=True)
    # Sort by atom_ranks
    next_nodes = next_nodes[np.argsort(atom_ranks[next_nodes])]
    # Filter visited nodes
    next_nodes = next_nodes[step_ids[next_nodes] == -1]

    # Iterate through neighbors
    while next_nodes.size > 0:  # If there are unvisited neighbors
        if len(next_nodes) == 1:  # Only one neighbor is unvisited
            next_node = next_nodes[0]  # Visit this neighbor
        else:
            alpha_1 = np.zeros_like(next_nodes, dtype=np.float32)
            # pylint: disable=unsupported-assignment-operation
            alpha_1[0] = 1.0
            alpha_2 = np.full_like(alpha_1, 1.0 / len(next_nodes))
            alpha = p * alpha_1 + (1.0 - p) * alpha_2
            next_node = np.random.choice(next_nodes, p=alpha)

        step_ids[next_node] = max(step_ids) + 1

        # Proceed to the next step
        step_ids = _traverse_scaffold(graph,
                                      atom_ranks,
                                      scaffold_nodes,
                                      p,
                                      next_node,
                                      step_ids)
        # Filter visited nodes
        next_nodes = next_nodes[step_ids[next_nodes] == -1]

    return step_ids


def _traverse_chain(graph: nx.Graph,
                    atom_ranks: np.ndarray,
                    scaffold_nodes: np.ndarray,
                    step_ids: np.ndarray,
                    p: float,
                    current_node: t.Optional[int] = None,
                    log_p=0.0):
    """
    An recursive function for stochastic traversal of side chains in `graph`

    Notes:
        The scaffold should be first traversed using `_traverse_scaffold`
    """
    # Initialize next_nodes
    # For the fist step
    if current_node is None:
        # Initialize next_nodes as an empty set
        next_nodes = set([])
        # Iterate through scaffold nodes
        for scaffold_node_id in scaffold_nodes:
            # Add all nodes directly connected to scaffold nodes
            next_nodes = next_nodes | set(graph.neighbors(scaffold_node_id))
        # Convert to ndarray
        next_nodes = np.array(list(next_nodes), dtype=np.int32)
        # Remove all scaffold nodes
        next_nodes = np.setdiff1d(next_nodes,
                                  scaffold_nodes,
                                  assume_unique=True)
    else:
        # Get neighbor nodes
        next_nodes = np.array(list(graph.neighbors(current_node)),
                              dtype=np.int32)
        # Remove all scaffold nodes
        next_nodes = np.setdiff1d(next_nodes,
                                  scaffold_nodes,
                                  assume_unique=True)
    # Sort by atom_ranks
    next_nodes = next_nodes[np.argsort(atom_ranks[next_nodes])]
    # Filter visited nodes
    next_nodes = next_nodes[step_ids[next_nodes] == -1]

    # Iterate through neighbors
    while next_nodes.size > 0:
        # If there are unvisited neighbors
        # Only one neighbor is unvisited
        if len(next_nodes) == 1:
            # Visit this neighbor
            next_node = next_nodes[0]
            log_p_step = 0.0
        else:
            alpha_1 = np.zeros_like(next_nodes, dtype=np.float32)
            # pylint: disable=unsupported-assignment-operation
            alpha_1[0] = 1.0
            alpha_2 = np.full_like(alpha_1, 1.0 / len(next_nodes))
            alpha = p * alpha_1 + (1.0 - p) * alpha_2
            next_node_index = np.random.choice(np.arange(len(next_nodes)),
                                               p=alpha)
            next_node = next_nodes[next_node_index]
            log_p_step = np.log(alpha[next_node_index])

        # # If scaffold have been iterated
        # if not is_scaffold_iteration:
        #     log_p += log_p_step

        log_p += log_p_step
        step_ids[next_node] = max(step_ids) + 1

        # Proceed to the next step
        step_ids, log_p = _traverse_chain(graph,
                                          atom_ranks,
                                          scaffold_nodes,
                                          step_ids,
                                          p,
                                          next_node,
                                          log_p)
        # Filter visited nodes
        next_nodes = next_nodes[step_ids[next_nodes] == -1]

    return step_ids, log_p
# !SECTION
