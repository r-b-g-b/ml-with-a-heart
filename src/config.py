import os.path as op

project_directory = op.abspath(op.join(__file__, op.pardir, op.pardir))
data_directory = op.join(project_directory, 'data')
raw_data_directory = op.join(data_directory, 'raw')
