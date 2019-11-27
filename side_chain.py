"""
Utility functions for analysing side chain distribution
"""
# region
import sys
# pylint: disable=unused-import
import time
import typing as t

import multiprocess as mp
import networkx as nx
from networkx.algorithms.components import connected_components
import numpy as np
import pandas as pd
from rdkit import Chem
# endregion


# SMARTS patterns for hydrogen bond donor and acceptors
# Adopted from the tutorials given by Daylight
HBA = '[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]'
HBA_QUERY = Chem.MolFromSmarts(HBA)
HBD = '[!$([#6,H0,-,-2,-3])]'
HBD_QUERY = Chem.MolFromSmarts(HBD)


class NoMatchException(Exception):
    """Cannot perform substructure match"""
    def __init__(self,
                 mol_smiles: str,
                 scaffold_smiles: str,
                 *args, **kwargs):
        """The initializer"""
        super(NoMatchException, self).__init__(*args, **kwargs)
        self.mol_smiles = mol_smiles
        self.scaffold_smiles = scaffold_smiles


class NoSubstitutionException(Exception):
    """Generated molecule does not have any substitution"""
    pass


def map_scaffold(mol: Chem.Mol, scaffold: Chem.Mol) -> t.Tuple[int]:
    """
    Get the location of a certain scaffold inside a molecule. The tuple retruned
    records the mapping from atom indices in scaffold `scaffold` to atom indices
    in molecule `mol`. Specifically, For atom `i` in scaffold `scaffold`,
    the value `return_val[i]` represents the index of the atom inside the molecule
    `mol`.
    """
    match = mol.GetSubstructMatch(scaffold)
    if not match:
        raise NoMatchException(Chem.MolToSmiles(mol),
                               Chem.MolToSmiles(scaffold))
    return match


def map_query(mol: Chem.Mol, query: Chem.Mol) -> t.Tuple[int]:
    """
    Get the set of indices of all atoms in molecule `mol` matching the query
    `query`
    """
    match = set()
    for match_i in mol.GetSubstructMatches(query):
        match = match | set(match_i)
    match = tuple(match)
    return match


def get_scaffold_anchors(mol: Chem.Mol,
                         scaffold_ids: t.Tuple[int]
                         ) -> t.Dict[int, int]:
    """
    Get the indices of atom directly connected with the scaffold. The
    dictionary returned maps the indices of directly connected atoms in the
    molecules to the index of the anchor atom in the scaffold
    """
    anchors = {}
    for scaffold_id, scaffold_id_in_mol in enumerate(scaffold_ids):
        scaffold_atom: Chem.Atom
        scaffold_atom = mol.GetAtomWithIdx(scaffold_id_in_mol)
        neighbor: Chem.Atom
        for neighbor in scaffold_atom.GetNeighbors():
            neighbor_id = neighbor.GetIdx()
            if neighbor_id not in anchors:
                anchors[neighbor_id] = scaffold_id
    return anchors


def convert_to_graph(mol: Chem.Mol,
                     scaffold_ids: t.Tuple[int],
                     anchors: t.Dict[int, int],
                     hba_ids: t.Tuple[int],
                     hbd_ids: t.Tuple[int]) -> nx.Graph:
    """
    Convert `Chem.Mol` object to `nx.Graph` object

    Args:
        mol (Chem.Mol):
            The molecule object to be converted
        scaffold_ids (t.Tuple[int]):
            The atom that corresponds to scaffolds
        anchors (t.Dict[int, int]):
            The mapping from atom in the molecule to atom in scaffold where it
            is attached to
        hba_ids (t.Tuple[int]):
            The atoms corresponding to hydrogen acceptors
        hbd_ids (t.Tuple[int]):
            The atoms corresponding to hydrogen donnors

    Returns:
        nx.Graph:
            The graph converted
    """
    # Initialize graph
    graph = nx.Graph()
    # Add nodes
    nodes = range(mol.GetNumAtoms())
    graph.add_nodes_from(nodes)
    # Add edges
    bond: Chem.Bond
    edges = [(bond.GetBeginAtomIdx(),
              bond.GetEndAtomIdx())
             for bond in mol.GetBonds()]
    graph.add_edges_from(edges)
    # Attach properties to nodes
    for node_id in nodes:
        atom_i: Chem.Atom = mol.GetAtomWithIdx(node_id)
        graph.nodes[node_id]['symbol'] = atom_i.GetSymbol()
    for node_id in anchors:
        graph.nodes[node_id]['anchor'] = anchors[node_id]
    for node_id in hba_ids:
        graph.nodes[node_id]['is_hba'] = True
    for node_id in hbd_ids:
        graph.nodes[node_id]['is_hbd'] = True
    for node_id in scaffold_ids:
        graph.nodes[node_id]['is_scaffold'] = True

    return graph


def propcess_graph(graph: nx.Graph) -> t.Tuple[pd.DataFrame, t.Dict[str, int]]:
    """
    Process information in graph, returning a table (table data + column names)
    as a result. Each row of the table represents a side-chain, and each column
    records one property of the side-chain, such as the number of heavy atoms it
    contains, whether it is a hydrogen bond donor or acceptor, ect.
    """
    scaffold_nodes = []
    for node_id, node in graph.nodes.items():
        if 'is_scaffold' in node:
            scaffold_nodes.append(node_id)
    # Remove scaffold
    graph.remove_nodes_from(scaffold_nodes)
    # Initialize data
    graph_info = []
    # Iterate through disconnected subgraphs
    for subgraph in connected_components(graph):
        attached_atom_id = None
        num_heavy_atoms, is_hbd, is_hba = 0, False, False
        for node_id in subgraph:
            node = graph.nodes[node_id]
            if attached_atom_id is None and 'anchor' in node:
                attached_atom_id = node['anchor']
            if not is_hba and 'is_hba' in node:
                is_hba = True
            if not is_hbd and 'is_hbd' in node:
                is_hbd = True
            num_heavy_atoms += 1
        if attached_atom_id is None:
            raise ValueError
        is_hbd_and_hba = is_hbd and is_hba
        graph_info.append([attached_atom_id,
                           num_heavy_atoms,
                           int(is_hbd),
                           int(is_hba),
                           int(is_hbd_and_hba)])
    # Convert graph_info to dataframe
    if not graph_info:
        raise NoSubstitutionException()
    graph_info = np.array(graph_info, dtype=np.int32)
    col_names = ['attached_atom_id',
                 'num_heavy_atoms',
                 'is_hbd',
                 'is_hba',
                 'is_hbd_and_hba']
    col_names = {key:val for val, key in enumerate(col_names)}
    return graph_info, col_names


def process_mol(mol: Chem.Mol,
                scaffold: Chem.Mol
                ) -> t.Tuple[pd.DataFrame, t.Dict[str, int]]:
    """
    Process molecule, returning a table (table data + column names)
    as a result. Each row of the table represents a side-chain, and each column
    records one property of the side-chain, such as the number of heavy atoms it
    contains, whether it is a hydrogen bond donor or acceptor, ect.
    """
    scaffold_map = map_scaffold(mol, scaffold)
    hba_ids, hbd_ids = map_query(mol, HBA_QUERY), map_query(mol, HBD_QUERY)
    anchors = get_scaffold_anchors(mol, scaffold_map)
    graph = convert_to_graph(mol, scaffold_map, anchors, hba_ids, hbd_ids)
    results = propcess_graph(graph)
    return results


def process_mol_set(smiles_loc: str,
                    output_loc: str,
                    scaffold_smiles: str):
    """Process a list of molecules

    Args:
        smiles_loc (str): The location of the file storing molecular SMILES
        output_loc (str): The location to save the result
        scaffold_smiles (str): The smiles of the scaffold
    """
    scaffold: Chem.Mol
    try:
        scaffold = Chem.MolFromSmiles(scaffold_smiles)
    except (ValueError, RuntimeError):
        raise ValueError(f'Invalid SMILES for scaffold: {scaffold_smiles}')
    if scaffold is None:
        raise ValueError(f'Invalid SMILES for scaffold: {scaffold_smiles}')

    # pylint: disable=no-member
    pool = mp.Pool(10)

    # The generator
    def _generate() -> t.Generator[str, None, None]:
        # pylint: disable=invalid-name
        with open(smiles_loc) as f:
            for line in f:
                yield line.rstrip()

    # The worker
    def _worker(_smiles: str) -> t.Optional[pd.DataFrame]:
        try:
            mol = Chem.MolFromSmiles(_smiles)
        except (ValueError, RuntimeError):
            return None
        if mol is None:
            return None

        try:
            return process_mol(mol, scaffold)
        except (NoMatchException, NoSubstitutionException):
            return None

    # Initialize the result table
    scaffold_size = scaffold.GetNumHeavyAtoms()
    col_names = ['num_subs',
                 'avg_size',
                 'num_hba',
                 'num_hbd',
                 'num_hbd_and_hba']
    results = np.zeros((scaffold_size, len(col_names)), dtype=np.float32)
    col_names = {key:val for val, key in enumerate(col_names)}

    # The iteration
    # pylint: disable=invalid-name
    g = _generate()
    for result_i in pool.imap(_worker, g, chunksize=100):
        if result_i is None:
            continue
        graph_info_i, col_names_i = result_i
        # Substuted locations
        loc = graph_info_i[:, col_names_i['attached_atom_id']]
        # Update avg_size
        mu_old = results[loc, col_names['avg_size']]
        mu_new = graph_info_i[:, col_names_i['num_heavy_atoms']]
        n_old = results[loc, col_names['num_subs']]
        results[loc, col_names['avg_size']] = \
            n_old / (n_old + 1) * mu_old + mu_new / (n_old + 1)
        # Update other variables
        results[loc, col_names['num_subs']] += 1
        results[loc, col_names['num_hba']] \
            += graph_info_i[:, col_names_i['is_hba']]
        results[loc, col_names['num_hbd']] \
            += graph_info_i[:, col_names_i['is_hbd']]
        results[loc, col_names['num_hbd_and_hba']] \
            += graph_info_i[:, col_names_i['is_hbd_and_hba']]

    results = pd.DataFrame(results, columns=list(col_names))
    results.to_csv(output_loc)


if __name__ == "__main__":
    process_mol_set(sys.argv[1],
                    sys.argv[2],
                    sys.argv[3])
