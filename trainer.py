#!/usr/bin/env python3.7
"""Model trainer"""
# region
import json
import math
import os
import sys
import time
import typing as t

import numpy as np
import torch
from torch import nn
from torch import optim

from data_utils import get_data_loader
from data_utils import get_data_loader_full
from mol_spec import MoleculeSpec
from deep_scaffold import DeepScaffold
# endregion


def _check_continuous(ckpt_loc: str) -> bool:
    """Check whether to continue the training or start a new session

    Args:
        ckpt_loc (str):
            The location of the checkpoint file

    Returns:
        bool:
            Whether to continue training
    """
    is_continuous = all([os.path.isfile(os.path.join(ckpt_loc, _n))
                         for _n in ['configs.json',
                                    'log.out',
                                    'mdl.ckpt',
                                    'optimizer.ckpt',
                                    'scheduler.ckpt']])
    return is_continuous


# pylint: disable=invalid-name
def _get_loader(network_loc: str,
                molecule_loc: str,
                exclude_ids_loc: str,
                split_by: str,
                batch_size: int,
                batch_size_test: int,
                num_iterations: int,
                num_workers: int,
                full: bool,
                training_only: bool,
                k: int,
                p: float,
                ms: MoleculeSpec
                ) -> t.Tuple[t.Iterable,
                             t.Optional[t.Iterable]]:
    """Helper function for getting data loaders

    Args:
        network_loc (str):
            Location of the bipartite network
        molecule_loc (str):
            Location of molecule SMILES strings
        exclude_ids_loc (str):
            The location storing the ids to be excluded from the training set
        split_by (str):
            Whether to split by scaffold or molecule
        batch_size (int):
            The batch size for training
        batch_size_test (int):
            The batch size for testing
        num_iterations (int):
            The number of total iterations for model training
        num_workers (int):
            The number of workers for loading dataset
        full (bool):
            Whether to use the full dataset for training
        training_only (bool):
            Only record training loss
        k (int):
            The number of importance samples
        p (float):
            The degree of stochasticity of importance sampling 0.0 for fully
            stochastic decoding, 1.0 for fully deterministic decoding
        ms (MoleculeSpec)

    Returns:
        t.Tuple[t.Iterable, t.Iterable]:
            DataLoaders for training and test data
    """
    if full:
        training_only = True
        loader_train = \
            get_data_loader_full(network_loc,
                                 molecule_loc,
                                 batch_size,
                                 num_iterations,
                                 num_workers,
                                 k, p, ms)
        loader_test = None
    else:
        loader_train, loader_test = \
            get_data_loader(network_loc,
                            molecule_loc,
                            exclude_ids_loc,
                            split_by,
                            batch_size,
                            batch_size_test,
                            num_iterations,
                            num_workers,
                            k, p, ms)
        if training_only:
            loader_test = None
    return loader_train, loader_test


def _init_mdl(num_atom_embedding: int,
              causal_hidden_sizes: t.Iterable,
              num_bn_features: int,
              num_k_features: int,
              num_layers: int,
              num_output_features: int,
              efficient: bool,
              activation: str,
              gpu_ids: t.Mapping[int, int]) -> nn.DataParallel:
    """Helper function for initializing model

    Args:
        num_atom_embedding (int):
            The size of the initial node embedding
        causal_hidden_sizes (tuple[int] or list[int]):
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
            Whether to use the memory efficient implementation of densenet

    Returns:
        nn.DataParallel:
            The model intialized
    """
    # SECTION Build model
    # SECTION Create empty model with config
    configs = {
        'num_atom_embedding': num_atom_embedding,
        'causal_hidden_sizes': causal_hidden_sizes,
        'num_bn_features': num_bn_features,
        'num_k_features': num_k_features,
        'num_layers': num_layers,
        'num_output_features': num_output_features,
        'efficient': efficient,
        'activation': activation,
    }
    mdl = DeepScaffold(**configs)
    # !SECTION

    # SECTION Weight initializer
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1.0)
    mdl.apply(init_weights)
    # !SECTION

    # SECTION Wrap into nn.DataParallel and move to gpu
    mdl = nn.DataParallel(mdl, device_ids=gpu_ids)
    mdl.cuda()
    # !SECTION
    # !SECTION
    return mdl


# pylint: disable=protected-access
def _restore(mdl: nn.Module,
             optimizer: optim.Optimizer,
             scheduler: optim.lr_scheduler._LRScheduler,
             ckpt_loc: str) -> t.Tuple[nn.Module,
                                       optim.Optimizer,
                                       optim.lr_scheduler._LRScheduler,
                                       int, float]:
    """Restore model training state

    Args:
        mdl (nn.Module):
            The randomly initialized model
        optimizer (optim.Optimizer):
            The optimizer
        scheduler (optim.lr_scheduler._LRScheduler):
            The scheduler for learning rate
        ckpt_loc (str):
            Location to store model checkpoints

    Returns:
        t.Tuple[nn.Module,
                optim.Optimizer,
                optim.lr_scheduler._LRScheduler,
                int, float]:
            The restored status
    """
    # Restore model checkpoint
    mdl.load_state_dict(
        torch.load(os.path.join(ckpt_loc, 'mdl.ckpt')))
    optimizer.load_state_dict(
        torch.load(os.path.join(ckpt_loc, 'optimizer.ckpt')))
    scheduler.load_state_dict(
        torch.load(os.path.join(ckpt_loc, 'scheduler.ckpt')))

    # Restore timer and step counter
    with open(os.path.join(ckpt_loc, 'log.out')) as f:
        records = f.readlines()
        if records[-1] != 'Training finished\n':
            final_record = records[-1]
        else:
            final_record = records[-2]
    global_counter, t_final = final_record.split('\t')[:2]
    global_counter = int(global_counter)
    t_final = float(t_final)
    t0 = time.time() - t_final * 60

    return mdl, optimizer, scheduler, global_counter, t0


def _save(mdl: nn.Module,
          optimizer: optim.Optimizer,
          scheduler: optim.lr_scheduler._LRScheduler,
          global_counter: int,
          t0: float,
          loss: float,
          current_lr: float,
          ckpt_loc: str) -> str:
    """Saving checkpoint to file

    Args:
        mdl (nn.Module):
            The randomly initialized model
        optimizer (optim.Optimizer):
            The optimizer
        scheduler (optim.lr_scheduler._LRScheduler):
            The scheduler for learning rate
        global_counter (int):
            The global counter for training
        t0 (float):
            The time training was started
        loss (float):
            The loss of the model
        current_lr (float):
            The current learning rate
        ckpt_loc (str):
            Location to store model checkpoints

    Return:
        str:
            The message string
    """
    # Save status
    torch.save(mdl.state_dict(),
               os.path.join(ckpt_loc, 'mdl.ckpt'))
    torch.save(optimizer.state_dict(),
               os.path.join(ckpt_loc, 'optimizer.ckpt'))
    torch.save(scheduler.state_dict(),
               os.path.join(ckpt_loc, 'scheduler.ckpt'))

    message_str = (f'{global_counter}\t'
                   f'{float(time.time() - t0) / 60}\t'
                   f'{loss}\t'
                   f'{current_lr}\n')
    return message_str


def _loss(mol_array: np.ndarray,
          log_p: np.ndarray,
          mdl: t.Union[nn.DataParallel,
                       DeepScaffold],
          device: torch.device,
          ) -> torch.Tensor:
    """A helper function for getting loss

    Args:
        mol_array (torch.Tensor):
            The molecule array to calculate the likelihood
        log_p (torch.Tensor):
            The log-likelihood value of each trajectory
        mdl (t.Union[nn.DataParallel,DeepScaffold]):
            The model

    Returns:
        torch.Tensor:
            The calculated loss
    """
    # Move to gpu
    # pylint: disable=not-callable
    mol_array = torch.tensor(mol_array,
                             dtype=torch.long,
                             device=device)
    log_p = torch.tensor(log_p,
                         dtype=torch.float32,
                         device=device)
    # Get shape and device information
    batch_size, k = log_p.shape
    # Flatten the first (batch_size) and the second (k) dimension
    mol_array = mol_array.view(batch_size * k, -1, 5)
    # Shuffle
    shuffle = torch.randperm(batch_size * k,
                             dtype=torch.long,
                             device=device)
    mol_array = mol_array[shuffle, ...]
    # Get likelihood
    # shape: batch_size * k
    ll = mdl(mol_array).sum(-1)
    # Shuffle back
    unshuffle = torch.argsort(shuffle)
    ll = ll[unshuffle]
    # Unflatten
    ll = ll.view(batch_size, k)
    # Get total likelihood
    ll = ll.sub(log_p)\
           .logsumexp(dim=-1)\
           .sub(math.log(float(k)))\
           .mean()
    # Get final loss
    loss = -ll
    return loss


# pylint: disable=protected-access
def _train_step(mdl: nn.Module,
                optimizer: optim.Optimizer,
                scheduler: optim.lr_scheduler._LRScheduler,
                min_lr: float,
                clip_grad: float,
                device: torch.device,
                iter_train: t.Iterator):
    """Helper function to perform one step of training

    Args:
         mdl (nn.Module):
            The randomly initialized model
        optimizer (optim.Optimizer):
            The optimizer
        scheduler (optim.lr_scheduler._LRScheduler):
            The scheduler for learning rate
        min_lr (float):
            The minimum learning rate
        clip_grad (float):
            Gradient clipping
        device (torch.device):
            The device where tensors should be intialized
        iter_train (t.Iterator):
            The iterator for trainer
    """
    # Prepare for training
    optimizer.zero_grad()  # Clear gradient
    if all([params_group['lr'] > min_lr
            for params_group in optimizer.param_groups]):
        # Update learning rate if it is still larger than min_lr
        scheduler.step()

    # Get data
    mol_array, log_p = next(iter_train)
    loss = _loss(mol_array, log_p, mdl, device)
    loss.backward()

    # Clip gradient
    torch.nn.utils.clip_grad_value_(mdl.parameters(), clip_grad)

    optimizer.step()
    return loss


# pylint: disable=protected-access
def _test_step(mdl: nn.Module,
               device: torch.device,
               iter_test: t.Iterator) -> float:
    """Helper function to perform one step of training

    Args:
        mdl (nn.Module):
            The randomly initialized model
        iter_test (t.Iterator):
            The iterator for tester
    """
    with torch.no_grad():
        mol_array, log_p = next(iter_test)
        mdl.eval()
        # Get loss
        loss = _loss(mol_array, log_p, mdl, device)
        mdl.train()
    return loss


# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=invalid-name
def engine(ckpt_loc: str = 'ckpt/ckpt-default',
           molecule_loc: str = 'data_utils/molecules.smi',
           network_loc: str = 'data_utils/scaffolds_molecules.pkl.gz',
           exclude_ids_loc: str = 'ckpt/ckpt-default/exclude_ids.txt',
           full: bool = False,
           split_by: str = 'molecule',
           training_only: bool = False,
           num_workers: int = 2,
           num_atom_embedding: int = 16,
           causal_hidden_sizes: t.Iterable = (32, 64),
           num_bn_features: int = 96,
           num_k_features: int = 24,
           num_layers: int = 20,
           num_output_features: int = 256,
           efficient: bool = False,
           ms: MoleculeSpec = MoleculeSpec.get_default(),
           activation: str = 'elu',
           lr: float = 1e-3,
           decay: float = 0.01,
           decay_step: int = 100,
           min_lr: float = 5e-5,
           summary_step: int = 200,
           clip_grad: float = 3.0,
           batch_size: int = 128,
           batch_size_test: int = 256,
           num_iterations: int = 50000,
           k: int = 5,
           p: float = 0.5,
           gpu_ids: t.Mapping[int, int] = (0, 1, 2, 3)):
    """
    Engine for training scaffold based VAE

    Args:
        ckpt_loc (str):
            Location to store model checkpoints
        molecule_loc (str):
            Location of molecule SMILES strings
        network_loc (str):
            Location of the bipartite network
        exclude_ids_loc (str):
            The location storing the ids to be excluded from the training set
        full (bool):
            Whether to use the full dataset for training, default to False
        split_by (str):
            Whether to split by scaffold or molecule
        training_only (str):
            Recording only training loss, default to False
        num_workers (int):
            Number of workers used during data loading, default to 1
        num_atom_embedding (int):
            The size of the initial node embedding
        causal_hidden_sizes (tuple[int] or list[int]):
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
            Whether to use the memory efficient implementation of densenet
        ms (mol_spec.MoleculeSpec)
        activation (str):
            The activation function used, default to 'elu'
        lr (float):
            (Initial) learning rate
        decay (float):
            The rate of learning rate decay
        decay_step (int):
            The interval of each learning rate decay
        min_lr (float):
            The minimum learning rate
        summary_step (int):
            Interval of summary
        clip_grad (float):
            Gradient clipping
        batch_size (int):
            The batch size for training
        batch_size_test (int):
            The batch size for testing
        num_iterations (int):
            The number of total iterations for model training
        k (int):
            The number of importance samples
        p (float):
            The degree of stochasticity of importance sampling 0.0 for fully
            stochastic decoding, 1.0 for fully deterministic decoding
        gpu_ids (tuple[int] or list[int]):
            Which GPUs are used for training
    """
    # ANCHOR Check whether to continue training
    is_continuous = _check_continuous(ckpt_loc)

    # ANCHOR Create iterators for training and test dataset
    loader_train, loader_test = _get_loader(network_loc,
                                            molecule_loc,
                                            exclude_ids_loc,
                                            split_by,
                                            batch_size,
                                            batch_size_test,
                                            num_iterations,
                                            num_workers,
                                            full,
                                            training_only,
                                            k, p, ms)
    iter_train = iter(loader_train)
    iter_test = iter(loader_test) if loader_test is not None else None

    # ANCHOR Initialize model with random params
    mdl = _init_mdl(num_atom_embedding,
                    causal_hidden_sizes,
                    num_bn_features,
                    num_k_features,
                    num_layers,
                    num_output_features,
                    efficient,
                    activation,
                    gpu_ids)

    # ANCHOR Initialize optimizer and scheduler
    optimizer = optim.Adam(mdl.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, decay_step, 1.0 - decay)

    # ANCHOR Load previously stored states
    if is_continuous:
        # restore states
        (mdl,
         optimizer,
         scheduler,
         t0,
         global_counter) = _restore(mdl,
                                    optimizer,
                                    scheduler,
                                    ckpt_loc)
    else:
        t0 = time.time()
        global_counter = 0
    device = torch.device(f'cuda:{gpu_ids[0]}')

    with open(os.path.join(ckpt_loc, 'log.out'),
              mode='a' if is_continuous else 'w') as f:
        if not is_continuous:
            f.write('global_step\ttime(min)\tloss\tlr\n')

        try:
            while True:
                global_counter += 1  # Update global counter
                # Perform one-step of training
                loss = _train_step(mdl,
                                   optimizer,
                                   scheduler,
                                   min_lr,
                                   clip_grad,
                                   device,
                                   iter_train)
                if global_counter % summary_step == 0:
                    if not training_only:
                        try:
                            loss = _test_step(mdl, device, iter_test)
                        except StopIteration:
                            iter_test = iter(loader_test)
                            loss = _test_step(mdl, device, iter_test)
                    loss = loss.item()

                    # Get learning rate
                    current_lr = [params_group['lr']
                                  for params_group
                                  in optimizer.param_groups][0]

                    # Save status
                    message_str = _save(mdl,
                                        optimizer,
                                        scheduler,
                                        global_counter,
                                        t0, loss, current_lr,
                                        ckpt_loc)

                    f.write(message_str)
                    f.flush()
        except StopIteration:
            if not training_only:
                try:
                    loss = _test_step(mdl, device, iter_test)
                except StopIteration:
                    iter_test = iter(loader_test)
                    loss = _test_step(mdl, device, iter_test)
            loss = loss.item()

            # Get learning rate
            current_lr = [params_group['lr']
                          for params_group in optimizer.param_groups][0]

            # Save status
            message_str = _save(mdl,
                                optimizer,
                                scheduler,
                                global_counter,
                                t0, loss, current_lr,
                                ckpt_loc)

            f.write(message_str)
            f.flush()
        f.write('Training finished')


def main(ckpt_loc):
    """Program entrypoint"""
    with open(os.path.join(ckpt_loc, 'config.json')) as f:
        config = json.load(f)
    config['ckpt_loc'] = ckpt_loc
    engine(**config)


if __name__ == '__main__':
    main(sys.argv[1])
