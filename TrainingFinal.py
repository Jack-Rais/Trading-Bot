import os
import json
import keras
import tensorflow as tf
import pandas as pd

from data import RetrieveDataset
from model import FinanceModel
from model.utils import ModelCheckpoint




dataset_filepaths = tf.io.gfile.glob(rf'dataset/*.tfrecord')
settings_filepath = 'settings.json'

model_settings_filepath = 'model_settings.json'
model_filepath = 'FinanceModel.weights.h5'
validation_split = 0.2

batch_size = 1
shuffle_buffer = 10

steps_per_epoch = 100
epochs = 1



retrieve = RetrieveDataset(dataset_filepaths, settings_filepath)

def mapping(x):
    return (x['titolo'], x['paragrafi'], x['prices']), x['label']

dataset = retrieve.get_dataset()
dataset = dataset.map(mapping).batch(batch_size).prefetch(tf.data.AUTOTUNE).shuffle(shuffle_buffer)

total_size = retrieve.get_size()
val_size = int(validation_split * total_size)

train_dataset = dataset.skip(val_size)
val_dataset = dataset.take(val_size)


with open(model_settings_filepath, 'r') as file:
    settings = json.load(file)

model = FinanceModel(**settings)

if os.path.exists(model_filepath):
    print('Loading weights...')
    model.load_weights(model_filepath)

model.compile(
    optimizer = keras.optimizers.Adam(),
    loss = keras.losses.BinaryCrossentropy()
)

checkpoint_callback = ModelCheckpoint(
    filepath = model_filepath, 
    monitor = 'val_loss', 
    save_weights_only = True,
    save_best_only = True, 
    mode = 'min',
    verbose = 1
)

history = model.fit(train_dataset, 
                    epochs = epochs + checkpoint_callback.epoch,
                    validation_data = val_dataset, 
                    callbacks = [
                        checkpoint_callback,
                        keras.callbacks.TensorBoard()
                    ],
                    steps_per_epoch = steps_per_epoch,
                    initial_epoch = checkpoint_callback.epoch
                    )

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()