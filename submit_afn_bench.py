"""
Submits jobs to create NAS-Bench-AFN
"""
import argparse
import sys
import wandb

import os
import sys
sys.path.append(os.path.join('/', 'home', 'garrett', 'workspace'))

from notferratu.activations.dag import BINARY_FUNCTIONS, UNARY_FUNCTIONS
from train_script import parse_arguments, run_training_by_args

ALL_FUNCTIONS = []
for b in BINARY_FUNCTIONS:
    for u1 in UNARY_FUNCTIONS:
        for u2 in UNARY_FUNCTIONS:
            fn = f'{b}({u1}(x),{u2}(x))'
            ALL_FUNCTIONS.append(fn)

WANDB_GROUP = 'mobilevit-v2-050-pangaea-search-space'

runs = wandb.Api(timeout=999999).runs('bingham/afn-bench', filters={'group': WANDB_GROUP})
already_evaluated_fns = [r.config['activation_fn'] for r in runs if r.state == 'finished' or r.state == 'running']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-idx', type=int, required=False, default=0,
        help='The index of the first function to evaluate')
    parser.add_argument('--end-idx', type=int, required=False, default=len(ALL_FUNCTIONS),
        help='The index of the last function to evaluate')
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]

    client_args = ['-d', 'imagenette',
                   '-m', 'mobilevit.MobileViT_V2_050',
                   '-p', 'AdamW',
                   '--wandb-project', 'afn-bench',
                   '--wandb-group', WANDB_GROUP]

    for i in range(args.start_idx, args.end_idx):
        fn = ALL_FUNCTIONS[i]
        if fn not in already_evaluated_fns:
            print(f'\nFunction {i}: {fn}')
            parsed_args = parse_arguments(client_args + ['--activation-fn', fn])
            run_training_by_args(parsed_args)
        else:
            print(f'\nFunction {i}: {fn} already evaluated')

if __name__ == '__main__':
    main()
