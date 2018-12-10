# -*- coding: utf-8 -*-
import os.path as op
import logging
import joblib
import numpy as np
from sklearn import linear_model
from .. import config


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    Cs = np.logspace(-3, 5, 20)
    logistic_regression = linear_model.LogisticRegressionCV(Cs=Cs, cv=5,
                                                            solver='lbfgs')
    logger.info('loading training data')
    X = np.load(op.join(config.data_directory, 'processed', 'X_train.npy'))
    y = np.load(op.join(config.data_directory, 'processed', 'y_train.npy'))

    logger.info('training the model')
    _ = logistic_regression.fit(X, y)

    logger.info('saving the model')
    model_path = op.join(config.project_directory, 'models',
                         'classifier.joblib')
    with open(model_path, 'wb') as f:
        joblib.dump(logistic_regression, f)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
