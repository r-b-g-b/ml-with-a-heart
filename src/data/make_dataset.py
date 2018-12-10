# -*- coding: utf-8 -*-
import os.path as op
import click
import logging
import numpy as np
import joblib
from .. import config
from ..data import load
from ..features import build_features

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    logger.info(config.data_directory)

    train_values, train_labels = load.load_train_data()
    preprocessors = build_features.build_preprocessors(train_values)
    train_features = build_features.preprocess_data(train_values, preprocessors)
    concatenator = build_features.ConcatenateFeatures()

    preprocessor_path = op.join(config.project_directory, 'models',
                                'preprocessors.joblib')
    with open(preprocessor_path, 'wb') as f:
        joblib.dump(preprocessors, f)

    _ = concatenator.fit(train_features, names=train_values.columns)
    X = concatenator.transform(train_features)
    np.save(op.join(config.data_directory, 'processed', 'X_train.npy'), X)
    np.save(op.join(config.data_directory, 'processed', 'y_train.npy'),
            train_labels.values.ravel())


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
