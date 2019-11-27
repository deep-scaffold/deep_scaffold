"""
Evaluating a single scaffold
"""
# region
import json
import os
import sys

import multiprocess as mp

import eval_sys
# endregion

def main(ckpt_loc: str):
    """
    The entrypoint of the program

    Args:
        ckpt_loc (str):
            The location of model checkpoints
    """
    # pylint: disable=no-member
    pool = mp.Pool(10)

    def mapper(*arg, **kwargs):
        return pool.map(*arg, **kwargs, chunksize=100)

    # Load generated samples
    # pylint: disable=invalid-name
    with open(os.path.join(ckpt_loc, 'samples.smi')) as f:
        smiles_g = list(map(lambda _x: _x.rstrip(), f))[:10000]
    # Load test set samples
    with open(os.path.join(ckpt_loc, 'exclude.smi')) as f:
        smiles_d = list(map(lambda _x: _x.rstrip(), f))
    # Get diversity and MMD
    (diversity_g,
     diversity_d,
     mmd) = eval_sys.get_mmd(smiles_g, smiles_d,
                             mapper=mapper)
    # Get property statistics
    prop_stat_g = eval_sys.get_prop_stat(smiles_g, mapper=mapper)
    prop_stat_d = eval_sys.get_prop_stat(smiles_d, mapper=mapper)

    results = {
        'generated':{
            'diversity': diversity_g,
            'mw': {
                'mean': prop_stat_g[0, 0],
                'std': (prop_stat_g[1, 0]) ** 0.5
            },
            'log_p': {
                'mean': prop_stat_g[0, 1],
                'std': (prop_stat_g[1, 1]) ** 0.5
            },
            'qed': {
                'mean': prop_stat_g[0, 2],
                'std': (prop_stat_g[1, 2]) ** 0.5
            }
        },
        'real':{
            'diversity': diversity_d,
            'mw': {
                'mean': prop_stat_d[0, 0],
                'std': (prop_stat_d[1, 0]) ** 0.5
            },
            'log_p': {
                'mean': prop_stat_d[0, 1],
                'std': (prop_stat_d[1, 1]) ** 0.5
            },
            'qed': {
                'mean': prop_stat_d[0, 2],
                'std': (prop_stat_d[1, 2]) ** 0.5
            }
        },
        'mmd': mmd
    }

    with open(os.path.join(ckpt_loc, 'stat.json'), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main(sys.argv[1])
