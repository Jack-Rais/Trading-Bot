from data.Data import GenerateDataset
from datetime import datetime
import os

API_KEY = 'PKPS66ILZGAOG7YTVGWB'
API_SECRET = 'gJc3XcPHWsD3kEMjBDRMeoRIJpswshUnSAriAn4I'

size_dataset = 100
symbol = 'AAPL'

generate = GenerateDataset(
    API_KEY,
    API_SECRET,
    datetime(2024, 10, 4),
    datetime(2017, 1, 1),
    use_first_date = True
)

generate.save_dataset_by_num(
    os.path.join('dataset', f'{symbol}_{generate.date.date()}_{size_dataset}.tfrecord'),
    symbol,
    size_dataset,
    save_last_date = True
)

'''from data.Data import RetrieveDataset
import tensorflow as tf
import os

retrieve = RetrieveDataset(
    tf.io.gfile.glob(rf'dataset/*.tfrecord')
)

dataset = retrieve.get_dataset()
print(retrieve.settings_filepath)
print(retrieve.get_size())'''

