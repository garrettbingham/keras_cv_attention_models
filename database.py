"""
Database subclass for searching for activation functions with ImageNet models.
"""

import numpy as np
import os
import sys
import tensorflow as tf

from keras_cv_attention_models.imagenet import data, train_func, losses

# Hacky path manipulation.  Eventually we should make it a proper package.
WORKSPACE_PATH = os.path.join('/', 'home', 'garrett', 'workspace')
sys.path.append(WORKSPACE_PATH)
from afn_bench.database import ActivationFunctionDatabase


class MobileViT_V2_050_ImageNet_AFD(ActivationFunctionDatabase):
    """
    A database of activation functions for the MobileViTv2-0.5 architecture trained on ImageNet.
    """
    def __init__(self, db_path):
        super().__init__(db_path)

        tf.random.set_seed(42)
        np.random.seed(42)

        # Most of these are default values from kecam train_script.py
        train_dataset, test_dataset, total_images, num_classes, steps_per_epoch = data.init_dataset(
            data_name='imagenet2012',
            input_shape=(160, 160, 3),
            # the batch size was actually 256, but we have to reduce it here to account for 
            # the extra memory usage from exposing multiple model outputs
            batch_size=128, 
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

        self.samples, self.labels = next(iter(train_dataset))
        self.loss = None # This will be set in the build_model function

        self.weights_per_layer = [
            432, 512, 288, 1024, 2048, 576, 4096, 8192, 1152, 8192, 8192, 1152, 16384, 1152, 8192,
            8256+129, 4096+64, 8192+128, 8192+64, 8256+129, 4096+64, 8192+128, 8192+64, 8192, 32768,
            2304, 49152, 1728, 18432, 18528+193, 9216+96, 18432+192, 18432+96, 18528+193, 9216+96,
            18432+192, 18432+96, 18528+193, 9216+96, 18432+192, 18432+96, 18528+193, 9216+96,
            18432+192, 18432+96, 18432, 73728, 3456, 98304, 2304, 32768, 32896+257, 16384+128,
            32768+256, 32768+128, 32896+257, 16384+128, 32768+256, 32768+128, 32896+257, 16384+128,
            32768+256, 32768+128, 32768, 2560+10
        ]

        # Evaluate these before beginning the search
        self.baseline_fns = [
            'elu(x)',
            'relu(x)',
            'selu(x)',
            'sigmoid(x)',
            'softplus(x)',
            'softsign(x)',
            'swish(x)',
            'tanh(x)',
        ]

        # Insert baseline functions if they don't exist
        self.cursor.execute(
            'SELECT fn_name FROM activation_functions WHERE fn_name IN ({})'.format(
                ','.join(['?'] * len(self.baseline_fns))
            ),
            self.baseline_fns
        )
        baseline_fns_in_db = [row[0] for row in self.cursor.fetchall()]
        # Insert the missing baseline functions
        missing_baseline_fns = list(set(self.baseline_fns) - set(baseline_fns_in_db))
        if len(missing_baseline_fns) > 0:
            self.populate_database(fn_names_list=missing_baseline_fns)
            self.calculate_fisher_eigs(fn_names_list=missing_baseline_fns)


    def build_model(self, fn_name):
        model = train_func.init_model(
            model='mobilevit.MobileViT_V2_050',
            input_shape=(160, 160, 3),
            num_classes=1000,
            autoinit=False, # AutoInit was not used in mobilevit experiments
            activation=fn_name,
        )

        self.loss, loss_weights, metrics = train_func.init_loss(
            bce_threshold=0.2,
            model_output_names=model.output_names
        )

        model = train_func.compile_model(
            model=model,
            optimizer='AdamW',
            lr_base=8e-3 * 256 / 512, # incorrect kecam comment; this should have been 8e-3 * 512 / 256, but it likely doesn't matter
            weight_decay=0.02,
            loss=self.loss,
            loss_weights=loss_weights,
            metrics=metrics,
            momentum=0.9,
        )

        return model


class AotNet50V2_ImageNet_AFD(ActivationFunctionDatabase):
    """
    A database of activation functions for the AotNet50V2 architecture trained on ImageNet.
    """
    def __init__(self, db_path):
        super().__init__(db_path)

        tf.random.set_seed(42)
        np.random.seed(42)

        # Most of these are default values from kecam train_script.py
        train_dataset, test_dataset, total_images, num_classes, steps_per_epoch = data.init_dataset(
            data_name='imagenet2012',
            input_shape=(160, 160, 3),
            # the batch size was actually 512*2, but we have to reduce it here to account for 
            # the extra memory usage from exposing multiple model outputs, and for using one GPU to
            # calculate FIM even though training uses four
            batch_size=128, 
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

        self.samples, self.labels = next(iter(train_dataset))
        self.loss = None # This will be set in the build_model function

        self.weights_per_layer = [
            9408, 4096, 36864, 16384, 16384, 16384, 36864, 16384, 16384, 36864, 16384, 32768,
            147456, 131072, 65536, 65536, 147456, 65536, 65536, 147456, 65536, 65536, 147456,
            65536, 131072, 589824, 524288, 262144, 262144, 589824, 262144, 262144, 589824, 262144,
            262144, 589824, 262144, 262144, 589824, 262144, 262144, 589824, 262144, 524288, 2359296,
            2097152, 1048576, 1048576, 2359296, 1048576, 1048576, 2359296, 1048576, 2048000+1000
        ]

        # Evaluate these before beginning the search
        self.baseline_fns = [
            'elu(x)',
            'relu(x)',
            'selu(x)',
            'sigmoid(x)',
            'softplus(x)',
            'softsign(x)',
            'swish(x)',
            'tanh(x)',
        ]

        # Insert baseline functions if they don't exist
        self.cursor.execute(
            'SELECT fn_name FROM activation_functions WHERE fn_name IN ({})'.format(
                ','.join(['?'] * len(self.baseline_fns))
            ),
            self.baseline_fns
        )
        baseline_fns_in_db = [row[0] for row in self.cursor.fetchall()]
        # Insert the missing baseline functions
        missing_baseline_fns = list(set(self.baseline_fns) - set(baseline_fns_in_db))
        if len(missing_baseline_fns) > 0:
            self.populate_database(fn_names_list=missing_baseline_fns)
            self.calculate_fisher_eigs(fn_names_list=missing_baseline_fns)


    def build_model(self, fn_name):
        model = train_func.init_model(
            model='aotnet.AotNet50V2',
            input_shape=(160, 160, 3),
            num_classes=1000,
            autoinit=True, # AutoInit was used in aotnet experiments
            activation=fn_name,
        )

        self.loss, loss_weights, metrics = train_func.init_loss(
            bce_threshold=0.2,
            model_output_names=model.output_names
        )

        model = train_func.compile_model(
            model=model,
            optimizer='AdamW',
            lr_base=8e-3 * 512 / 128,
            weight_decay=0.02,
            loss=self.loss,
            loss_weights=loss_weights,
            metrics=metrics,
            momentum=0.9,
        )

        return model
