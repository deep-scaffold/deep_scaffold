"""Module used to build test dataset"""
# region
import gc
import gzip
import io
import pickle
import sys
import typing as t
# endregion


def get_test_dataset(scaffold_network_loc: str,
                     exclude_ids_loc: str,
                     output_loc: str,
                     threshold: t.Tuple[int, int] = (10, 10000)):
    """
    Get test molecule-scaffold dataset using exclude_ids_loc

    Args:
        scaffold_network_loc (str):
            The location of scaffold_molecules.pkl.gz
        exclude_ids_loc (str):
            The files containing the molecule ids to be excluded
        output_loc (str):
            The location to output the result
        threshold (t.Tuple[int, int], optional):
            The minimum and maximum amount of molecule needs to be contained
            in the scaffold. Defaults to (10, 10000).
    """
    # Load dataset
    gc.disable()
    # pylint: disable=invalid-name
    with gzip.open(scaffold_network_loc, 'rb') as f:
        scaffold2molecule: t.Dict[int, t.Set[int]]
        # Compile the mapping between scaffold and molecule
        scaffold2molecule, _, _ = pickle.load(io.BufferedReader(f))
    gc.enable()

    # Load exclude ids
    exclude_ids = []
    with open(exclude_ids_loc) as f:
        for line in f:
            exclude_ids.append(int(line.rstrip()))
    exclude_ids = set(exclude_ids)

    # Initialize the test subset of the database
    scaffold2molecule_test = {}

    # Iterate over all scaffolds
    for scaffold_id in scaffold2molecule:
        # Get the intersection between the full dataset and the set exclude_ids
        molecule_ids = scaffold2molecule[scaffold_id] & exclude_ids
        if threshold[0] <= len(molecule_ids) <= threshold[1]:
            scaffold2molecule_test[scaffold_id] = molecule_ids
        else:
            continue

    # Save using pickle
    gc.disable()
    with gzip.open(output_loc, 'wb') as f:
        pickle.dump(scaffold2molecule_test,
                    io.BufferedWriter(f),
                    pickle.HIGHEST_PROTOCOL)
    gc.enable()


if __name__ == "__main__":
    get_test_dataset(sys.argv[1],
                     sys.argv[2],
                     sys.argv[3])
