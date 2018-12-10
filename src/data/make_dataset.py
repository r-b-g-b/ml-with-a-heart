# -*- coding: utf-8 -*-
import os.path as op
import logging
import urllib.request


DATA_URIS = {'train_values': 'https://s3.amazonaws.com/drivendata/data/54/public/train_values.csv',
             'train_labels': 'https://s3.amazonaws.com/drivendata/data/54/public/train_labels.csv',
             'test_values': 'https://s3.amazonaws.com/drivendata/data/54/public/test_values.csv'}

project_directory = op.join(op.dirname(__file__), op.pardir, op.pardir)
raw_data_directory = op.join(project_directory, 'data', 'raw')


def download_data():
    """Download data to local disk
    """
    for name, uri in DATA_URIS.items():
        path = op.join(raw_data_directory, name + '.csv')
        urllib.request.urlretrieve(uri, path)


def main():
    download_data()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
