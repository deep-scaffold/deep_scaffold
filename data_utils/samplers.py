"""Sampler for scaffold-molecule pair"""
# region
import random
import typing as t

from torch.utils import data

from data_utils.datasets import ScaffoldMolDataset
# endregion


__all__ = ['ScaffoldMolSampler']


class ScaffoldMolSampler(data.Sampler):
    """Batch sampler for scaffold-molecule pairs from the dataset"""

    def __init__(self,
                 dataset: ScaffoldMolDataset,
                 batch_size: t.Tuple[int, int],
                 num_iterations: int,
                 exclude_ids_loc: t.Optional[str] = None,
                 training: t.Optional[bool] = None,
                 split_type: t.Optional[str] = None):
        """
        Building the sampler

        Args:
            dataset (Dataset):
                The dataset to be sampled
            batch_size (tuple[int, int]):
                The batch sizes
            exclude_ids_loc (str or None):
                The location storing the ids to be excluded from the training
                set
            split_type (str or None):
                The type of splits
            training (bool or None):
                Whether the sampler is used for training data or for test data
            num_iterations (int):
                The number of total iterations
        """
        super(ScaffoldMolSampler, self).__init__(dataset)

        self.dataset = dataset
        self.batch_size_scaffold, self.batch_size_mol = batch_size
        self.num_iterations = num_iterations

        # Support three types of splits: scaffold, molecule and no split
        self.split_type = split_type
        assert self.split_type in [None, 'scaffold', 'molecule']

        self.training = training
        if self.split_type is not None:
            assert isinstance(self.training, bool)

        self.exclude_ids = []
        if self.split_type is not None:
            assert isinstance(exclude_ids_loc, str)
            # pylint: disable=invalid-name
            with open(exclude_ids_loc) as f:
                self.exclude_ids = [int(line.rstrip()) for line in f]
        self.exclude_ids_set = set(self.exclude_ids)

    # pylint: disable=too-many-branches
    def __iter__(self) -> t.Generator[t.List[t.Tuple[int, int, int]],
                                      None, None]:
        """
        Iterate through the dataset

        Yields:
            list[tuple[int, int, int]]:
                - scaffold_id (int): The index of the scaffold
                - molecule_id (int): The index of the molecule
                - record_id (int): The record id
        """
        shuffled_index_scaffold, shuffled_index_molecule = [], []

        for _ in range(self.num_iterations):
            # Sample by scaffold
            batch_scaffold = []

            while len(batch_scaffold) < self.batch_size_scaffold:
                # Make sure that the index are sufficient
                if not shuffled_index_scaffold:
                    num_scaffolds = len(self.dataset.scaffold_ids)
                    shuffled_index_scaffold = list(range(num_scaffolds))
                    random.shuffle(shuffled_index_scaffold)

                # Get the scaffold id
                scaffold_index = shuffled_index_scaffold.pop()
                scaffold_id = self.dataset.scaffold_ids[scaffold_index]

                # Continue if the scaffold id is to be excluded
                if self.split_type == 'scaffold':
                    if self.training == (scaffold_id in self.exclude_ids_set):
                        continue

                # Get the set of molecules corresponding to the scaffold
                molecule_ids = self.dataset[scaffold_id, None]  # type: set

                # Remove molecules to be excluded
                if self.split_type == 'molecule':
                    if self.training:
                        molecule_ids = molecule_ids - self.exclude_ids_set
                    else:
                        molecule_ids = molecule_ids & self.exclude_ids_set

                # Continue if the set is empty
                if not molecule_ids:
                    continue

                # Transform it to list
                molecule_ids = list(molecule_ids)

                # Random sample molecule id
                molecule_id = random.choice(molecule_ids)

                # Random sample record
                num_records = len(self.dataset[scaffold_id, molecule_id])
                record_id = random.choice(list(range(num_records)))

                # Append the result to batch
                batch_scaffold.append((scaffold_id, molecule_id, record_id))

            # Sample by molecule
            batch_molecule = []  # Initialize the batch

            while len(batch_molecule) < self.batch_size_mol:
                # Make sure that the index are sufficient
                if not shuffled_index_molecule:
                    num_mol = len(self.dataset.molecule_ids)
                    shuffled_index_molecule = list(range(num_mol))
                    random.shuffle(shuffled_index_molecule)

                # Get the molecule id
                molecule_index = shuffled_index_molecule.pop()
                molecule_id = self.dataset.molecule_ids[molecule_index]

                # Continue if the scaffold id is to be excluded
                if self.split_type == 'molecule':
                    if self.training == (molecule_id in self.exclude_ids_set):
                        continue

                # Get the set of scaffolds corresponding to the molecule
                scaffold_ids = self.dataset[None, molecule_id]  # type: set

                # Remove scaffolds to be excluded
                if self.split_type == 'scaffold':
                    if self.training:
                        scaffold_ids = scaffold_ids - self.exclude_ids_set
                    else:
                        scaffold_ids = scaffold_ids & self.exclude_ids_set

                # Continue if the set is empty
                if not scaffold_ids:
                    continue

                # Transform it to list
                scaffold_ids = list(scaffold_ids)

                # Random sample molecule id
                scaffold_id = random.choice(scaffold_ids)

                # Random sample record
                num_records = len(self.dataset[scaffold_id, molecule_id])
                record_id = random.choice(list(range(num_records)))

                # Append the result to batch
                batch_molecule.append((scaffold_id, molecule_id, record_id))

            batch = batch_scaffold + batch_molecule
            yield batch

    def __len__(self):
        return self.num_iterations
