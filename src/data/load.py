import os.path as op
import pandas as pd
from .. import config


def load_train_data():
    """Load the training datasets

    Returns
    -------
    train_values : pandas.core.frame.DataFrame
        A dataframe of training input variables
    train_labels : pandas.core.frame.DataFrame
        A dataframe of training output variables (i.e., targets)
    """
    path = op.join(config.raw_data_directory, 'train_values.csv')
    train_values = pd.read_csv(path, index_col='patient_id')

    path = op.join(config.raw_data_directory, 'train_labels.csv')
    train_labels = pd.read_csv(path, index_col='patient_id')

    return train_values, train_labels
