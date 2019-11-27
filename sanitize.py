"""Sanitizing SMILES"""
# region
import sys

from rdkit import Chem
# endregion


def main(file_loc: str):
    """
    The entrypoint for the program

    Args:
        file_loc (str):
            The location of the file to be processed
    """
    with open(file_loc) as f:
        with open(file_loc + '.bk', 'w') as h:
            for line in f:
                try:
                    mol = Chem.MolFromSmiles(line.rstrip())
                except ValueError:
                    continue
                if mol is None:
                    continue
                smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
                h.write(f'{smiles}\n')


if __name__ == "__main__":
    main(sys.argv[1])
