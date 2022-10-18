"""
Build the AFN-Bench dataset
"""
import os, sys
sys.path.append(os.path.join('/', 'home', 'garrett', 'workspace', 'notferratu'))
sys.path.append(os.path.join('/', 'home', 'garrett', 'workspace', 'keras_cv_attention_models'))

import argparse
import json
import matplotlib
matplotlib.rcParams['figure.dpi'] = 300
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import wandb

from keras_cv_attention_models.imagenet import data, train_func, losses
from utils.fisher import FIM

ALL_KEYS = [
    'W&B Run Config',
    # 'W&B Run Summary',
    'Train Accuracy',
    'Validation Accuracy',
    'Test Accuracy',
    'Train Loss',
    'Validation Loss',
    'Test Loss',
    'Runtime (s)',
    'Activation Function',
    'Fisher log eigenvalue CDF',
    # 'np.linspace(-5, 5, 100)',
    # 'eta',
    # 'eta_',
    # 'zeta',
]


def get_eigenvalue_cdf(afn_name, model_name):
    # Set seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    assert model_name == 'mobilevit.MobileViT_V2_050', 'Only MobileViT_V2_050 is supported'

    try:
        model = train_func.init_model(
            model=model_name,
            input_shape=(160, 160, 3),
            num_classes=10,
            autoinit=False, # AutoInit was not used in mobilevit experiments
            # activation_fn=afn_name,   # For some reason dL/dw is None when the model is reinstantiated after model_surgery
            activation=afn_name,        # so instead we just ensure the model has the right activation function at initialization
        )

        # Most of these are default values from kecam train_script.py
        train_dataset, test_dataset, total_images, num_classes, steps_per_epoch = data.init_dataset(
            data_name='imagenette',
            input_shape=(160, 160, 3),
            batch_size=128, # the batch size was actually 256, but we have to reduce it here to account for the extra memory usage from exposing multiple model outputs
            mixup_alpha=0.1,
            cutmix_alpha=1.0,
            rescale_mode='torch',
            eval_central_crop=0.95,
            random_crop_min=0.08,
            resize_method='bicubic',
            resize_antialias=True,
            random_erasing_prob=0,
            magnitude=6,
            num_layers=2,
            use_positional_related_ops=True,
            token_label_file=None,
            token_label_target_patches=-1,
            teacher_model=None,
            teacher_model_input_shape=-1,
            use_shuffle=False, # For reproducibility
        )

        loss, loss_weights, metrics = train_func.init_loss(
            bce_threshold=0.2,
            model_output_names=model.output_names
        )

        model = train_func.compile_model(
            model=model,
            optimizer='AdamW',
            lr_base=8e-3 * 256 / 512,
            weight_decay=0.02,
            loss=loss,
            loss_weights=loss_weights,
            metrics=metrics,
            momentum=0.9,
        )

        samples, labels = next(iter(train_dataset))
        fim = FIM(model, samples, labels, loss)
        eigenvalues = fim.calculate_eigenvalues(log_scale=True)
        print('Not flattening the eigenvalues!')
        # eigenvalues = np.array([e for eigs in eigenvalues for e in eigs]).flatten()

        BINS = np.linspace(-100, 100, 1000+1)
        pdf, _, _ = plt.hist(eigenvalues, bins=BINS, density=True)
        cdf = np.cumsum(pdf)
        cdf /= cdf[-1]

        # ndarray is not json serializable
        cdf = [c for c in cdf]

    except Exception as e:
        print(afn_name, e)
        cdf = None
        raise e

    return cdf


def nan_if_error(theory_fn, afn):
    try:
        return theory_fn(afn)
    except TypeError:
        return 'NaN'


def get_datum(run, afn_name, model_name, key):
    key_to_data = {
        'W&B Run Config'            : lambda: run.config,
        # 'W&B Run Summary'           : lambda: run.summary,
        'Train Accuracy'            : lambda: run.summary['acc'],
        'Validation Accuracy'       : lambda: run.summary['val_acc'],
        'Test Accuracy'             : lambda: run.summary['val_acc'],
        'Train Loss'                : lambda: run.summary['loss'],
        'Validation Loss'           : lambda: run.summary['val_loss'],
        'Test Loss'                 : lambda: run.summary['val_loss'],
        'Runtime (s)'               : lambda: run.summary['_runtime'],
        'Activation Function'       : lambda: run.config['activation_fn'],
        'Fisher log eigenvalue CDF' : lambda: get_eigenvalue_cdf(afn_name, model_name),
        # 'np.linspace(-5, 5, 100)'   : lambda: [float(x) for x in afn(np.linspace(-5, 5, 100)).numpy()],
        # 'eta'                       : lambda: nan_if_error(theory_eta, afn),
        # 'eta_'                      : lambda: nan_if_error(theory_eta_, afn),
        # 'zeta'                      : lambda: nan_if_error(theory_zeta, afn),
    }
    return key_to_data[key]()


def build_dataset(runs, data_path, model_name=None):
    # load JSON data from data_path if available
    try:
        with open(data_path, 'r') as f:
            dataset = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        dataset = {}

    # iterate over runs to build the dataset
    for run in tqdm(runs):
        afn_name = run.config['activation_fn']

        # fill in the missing data for each activation function
        afn_data = dataset.get(afn_name, {})
        something_to_update = False
        for key in ALL_KEYS:
            if key not in afn_data or afn_data[key] is None:
                afn_data[key] = get_datum(run, afn_name, model_name, key)
                something_to_update = True

        # save the dataset with each iteration
        dataset[afn_name] = afn_data
        if something_to_update:
            with open(data_path, 'w') as f:
                json.dump(dataset, f)

    return dataset


if __name__ == '__main__':
    # Decide which experiment we are gathering data for
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--batch', type=str, required=False,
        help='Batch of runs to process, expressed as a fraction like "3/8".')
    args = parser.parse_args()
    experiment = args.experiment
    if args.batch is not None:
        batch = int(args.batch.split('/')[0])
        num_batches = int(args.batch.split('/')[1])
    else:
        batch = 1
        num_batches = 1

    # Set experiment-specific parameters
    if 'mobilevit' in experiment:
        model_name = 'mobilevit.MobileViT_V2_050'
        wandb_group = 'mobilevit-v2-050-pangaea-search-space'
    
    else:
        raise ValueError('Unknown experiment: {}'.format(experiment))

    # Get JSON path and choose the keys we care about
    data_path = os.path.join(os.path.expanduser('~'), 'workspace', 'keras_cv_attention_models', 'afn_bench_data', f'{experiment}-{batch}-{num_batches}.json')

    # Get the W&B runs that finished successfully
    api = wandb.Api(timeout=99999)
    runs = api.runs('bingham/afn-bench', filters={'group' : f'{wandb_group}'})
    runs = [r for r in runs if r.state == 'finished' and 'acc' in r.summary]
    # Sort runs by name so that we can split them into batches
    runs = sorted(runs, key=lambda r: r.name)
    runs_to_process = runs[(batch-1)::num_batches]

    dataset = build_dataset(runs_to_process, data_path, model_name)
