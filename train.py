#! /usr/bin/python
# -*- coding: utf8 -*-
import os

import numpy as np
import tensorflow as tf

from deepsleep.trainer import DeepFeatureNetTrainer, DeepSleepNetTrainer
from deepsleep.sleep_stage import (NUM_CLASSES,
                                   EPOCH_SEC_LEN,
                                   SAMPLING_RATE)

import argparse
import os
import shutil


# Create an ArgumentParser instance
parser = argparse.ArgumentParser()

# Define command-line arguments
parser.add_argument('--data_dir', type=str, default='data', help='Directory where to load training data.')
parser.add_argument('--output_dir', type=str, default='output', help='Directory where to save trained models and outputs.')
parser.add_argument('--n_folds', type=int, default=20, help='Number of cross-validation folds.')
parser.add_argument('--fold_idx', type=int, default=0, help='Index of cross-validation fold to train.')
parser.add_argument('--pretrain_epochs', type=int, default=100, help='Number of epochs for pretraining DeepFeatureNet.')
parser.add_argument('--finetune_epochs', type=int, default=200, help='Number of epochs for fine-tuning DeepSleepNet.')
parser.add_argument('--resume', action='store_true', help='Whether to resume the training process.')

# Parse the command-line arguments
FLAGS = parser.parse_args()

# Access the parsed arguments
data_dir = FLAGS.data_dir
output_dir = FLAGS.output_dir
n_folds = FLAGS.n_folds
fold_idx = FLAGS.fold_idx
pretrain_epochs = FLAGS.pretrain_epochs
finetune_epochs = FLAGS.finetune_epochs
resume = FLAGS.resume

# Example usage of the parsed arguments
print("Data directory:", data_dir)
print("Output directory:", output_dir)
print("Number of folds:", n_folds)
print("Fold index:", fold_idx)
print("Pretrain epochs:", pretrain_epochs)
print("Finetune epochs:", finetune_epochs)
print("Resume:", resume)

def pretrain(n_epochs):
    trainer = DeepFeatureNetTrainer(
        data_dir=FLAGS.data_dir, 
        output_dir=FLAGS.output_dir,
        n_folds=FLAGS.n_folds, 
        fold_idx=FLAGS.fold_idx,
        batch_size=100, 
        input_dims=EPOCH_SEC_LEN*100, 
        n_classes=NUM_CLASSES,
        interval_plot_filter=50,
        interval_save_model=100,
        interval_print_cm=10
    )
    pretrained_model_path = trainer.train(
        n_epochs=n_epochs, 
        resume=FLAGS.resume
    )
    return pretrained_model_path


def finetune(model_path, n_epochs):
    trainer = DeepSleepNetTrainer(
        data_dir=FLAGS.data_dir, 
        output_dir=FLAGS.output_dir, 
        n_folds=FLAGS.n_folds, 
        fold_idx=FLAGS.fold_idx, 
        batch_size=10, 
        input_dims=EPOCH_SEC_LEN*100, 
        n_classes=NUM_CLASSES,
        seq_length=25,
        n_rnn_layers=2,
        return_last=False,
        interval_plot_filter=50,
        interval_save_model=100,
        interval_print_cm=10
    )
    finetuned_model_path = trainer.finetune(
        pretrained_model_path=model_path, 
        n_epochs=n_epochs, 
        resume=FLAGS.resume
    )
    return finetuned_model_path


def main(argv=None):
    # Output dir
    output_dir = os.path.join(FLAGS.output_dir, "fold{}".format(FLAGS.fold_idx))
    # if not FLAGS.resume:
    #     if tf.gfile.Exists(output_dir):
    #         tf.gfile.DeleteRecursively(output_dir)
    #     tf.gfile.MakeDirs(output_dir)
    if not FLAGS.resume:
        if os.path.exists(output_dir):
            print(f"Will delete {output_dir}")
            # shutil.rmtree(output_dir)
        print(f"And make {output_dir}")
        # os.makedirs(output_dir)

    pretrained_model_path = pretrain(
        n_epochs=FLAGS.pretrain_epochs
    )
    finetuned_model_path = finetune(
        model_path=pretrained_model_path, 
        n_epochs=FLAGS.finetune_epochs
    )


if __name__ == "__main__":
    #tf.compat.v1.app.run()
    main()
