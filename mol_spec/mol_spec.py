"""Specifications for atom types, bond types, etc."""
# region
import os

from rdkit import Chem
# endregion


__all__ = ['MoleculeSpec']


class MoleculeSpec:
    """A utility class for managing atom types and bond types"""
    def __init__(self,
                 file_name: str = os.path.join(os.path.dirname(__file__),
                                               'atom_types.txt'),
                 max_num_atoms: int = 50,
                 max_num_bonds: int = 100):
        """The helper class to store information about atom types and bond types

        Args:
            file_name:
                The location where atom type information and bond information
                is stored, default to 'datasets/atom_types.txt'
            max_num_atoms:
                The maximum allowed size of molecules measured using the number
                of heavy atoms, default to 50
            max_num_bonds:
                The maximum allowed size of molecules measured using the number
                of bonds, default to 100
        """
        self.atom_types = []
        self.atom_symbols = []
        # pylint: disable=invalid-name
        with open(file_name) as f:
            for line in f:
                atom_type_i = line.strip('\n').split(',')
                self.atom_types.append((atom_type_i[0],
                                        int(atom_type_i[1]),
                                        int(atom_type_i[2])))
                if atom_type_i[0] not in self.atom_symbols:
                    self.atom_symbols.append(atom_type_i[0])

        self.bond_orders = [Chem.BondType.AROMATIC,
                            Chem.BondType.SINGLE,
                            Chem.BondType.DOUBLE,
                            Chem.BondType.TRIPLE]

        self.max_num_atoms = max_num_atoms
        self.max_num_bonds = max_num_bonds

    def get_atom_type(self, atom: Chem.Atom) -> int:
        """
        Get the atom type (represented as the index in self.atom_types) of the
        given atom `atom`

        Args:
            atom (Chem.Atom):
                The input atom

        Returns:
            int:
                The atom type as int
        """
        atom_symbol = atom.GetSymbol()
        atom_charge = atom.GetFormalCharge()
        atom_hs = atom.GetNumExplicitHs()
        return self.atom_types.index((atom_symbol, atom_charge, atom_hs))

    def get_bond_type(self, bond: Chem.Bond) -> int:
        """
        Get the atom type (represented as the index in self.bond_types) of
        the given `bond`

        Args:
            bond (Chem.Bond):
                The input bond

        Returns:
            int:
                The bond type as int
        """
        return self.bond_orders.index(bond.GetBondType())

    def index_to_atom(self, idx: int) -> Chem.Atom:
        """ Create a new atom of type `idx`

        Args:
            idx (int):
                The type of atom to create

        Returns:
            Chem.Atom:
                The new atom created
        """
        atom_type = self.atom_types[idx]
        atom_symbol, atom_charge, atom_hs = atom_type
        atom = Chem.Atom(atom_symbol)
        atom.SetFormalCharge(atom_charge)
        atom.SetNumExplicitHs(atom_hs)
        return atom

    def index_to_bond(self,
                      mol: Chem.Mol,
                      begin_id: int,
                      end_id: int,
                      idx: int):
        """ Create a new bond in the given molecule `mol`

        Args:
            mol (Chem.Mol):
                The molecule where the new bond should be created
            begin_id, end_id (int):
                The (starting and terminating) location of the new bond
            idx (int):
                The type of the new bond
        """
        mol.AddBond(begin_id, end_id, self.bond_orders[idx])

    @property
    def num_atom_types(self) -> int:
        """The total number of atom types"""
        return len(self.atom_types)

    @property
    def num_bond_types(self) -> int:
        """The total number of bond types"""
        return len(self.bond_orders)

    # pylint: disable=undefined-variable
    @staticmethod
    def get_default():
        """Get default molecule spec"""
        return MoleculeSpec()
