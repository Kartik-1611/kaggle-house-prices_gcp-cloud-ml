import tensorflow as tf
import pandas as pd
import numpy as np
import os
import subprocess
import argparse
from . import model


WORKING_DIR = os.getcwd()
DATA_FILE_NAME = 'train.csv'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-dir',
        type=str,
        help='GCS location to write checkpoints and export models')
    parser.add_argument(
        '--train-file-path',
        type=str,
        required=True,
        help='Dataset file local or GCS')
    parser.add_argument(
        '--train-steps',
        type=float,
        default=1500,
        help='Number of times to go through the data, default=1500')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    args, _ = parser.parse_known_args()
    return args


def download_files_from_gcs(source, destination):
    local_file_name = destination
    gcs_input_path  = source

    local_file_path = os.path.join(WORKING_DIR, local_file_name)
    if gcs_input_path:
        subprocess.check_call(['gsutil', 'cp', gcs_input_path, local_file_path])

    return local_file_path


def normalize_df(df):

    df_stats = df.describe()
    df_stats = df_stats.transpose()

    return (df - df_stats['mean']) / df_stats['std']


def load_data(path='train.csv', test_split=0.2):

    assert 0 <= test_split < 1
    if not path:
        raise ValueError('No dataset file defined')

    if path.startswith('gs://'):
        download_files_from_gcs(path, destination=DATA_FILE_NAME)
        path = DATA_FILE_NAME

    columns = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath',
               'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'SalePrice']

    raw_train_data = pd.read_csv(path, usecols=columns)

    raw_train_data = raw_train_data.fillna(raw_train_data.median())

    train_dataset = raw_train_data.sample(frac=0.8, random_state=5)
    train_dataset = train_dataset.sort_index()
    test_dataset  = raw_train_data.drop(train_dataset.index)

    train_labels  = np.log(train_dataset['SalePrice'])
    train_dataset = train_dataset.drop(columns=['SalePrice'])

    test_labels   = np.log(test_dataset['SalePrice'])
    test_dataset  = test_dataset.drop(columns=['SalePrice'])

    normed_train_dataset = normalize_df(train_dataset)
    normed_test_dataset  = normalize_df(test_dataset)

    return (normed_train_dataset, train_labels), (normed_test_dataset, test_labels)


def train_and_evaluate(args):

    (train_data, train_labels), (test_data, test_labels) = load_data(path=args.train_file_path)

    features = []
    for col in train_data.columns:
        if col != 'SalePrice':
            features.append(tf.feature_column.numeric_column(col))

    train_input_fn = model.get_input_fn(train_data, train_labels, target_column='SalePrice', mode='train')
    eval_input_fn  = model.get_input_fn(test_data, test_labels, target_column='SalePrice', mode='eval')
    estimator      = model.get_estimator(features=features, model_dir=args.model_dir)

    estimator.train(input_fn=train_input_fn, steps=args.train_steps)
    estimator.evaluate(input_fn=eval_input_fn)

    # Export the model to GCS bucket
    estimator.export_savedmodel(export_dir_base=args.model_dir, export_input_fn=model.get_export_fn(features=features))


if __name__ == '__main__':
    args = get_args()
    tf.logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)

