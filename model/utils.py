import os
import keras
import pickle
import inspect

import tensorflow as tf 

def get_calling_file_directory():
    
    caller_frame = inspect.stack()[1]
    caller_filename = caller_frame.filename
    
    caller_directory = os.path.dirname(os.path.abspath(caller_filename))
    return caller_directory

class ModelCheckpoint(keras.callbacks.ModelCheckpoint):

    def __init__(self, filepath:str | os.PathLike,
                       monitor:str = 'val_loss',
                       verbose:int = 0,
                       save_best_only:bool = False,
                       save_weights_only:bool = False,
                       mode:str = 'auto',
                       save_freq:str | int = 'epoch',
                       use_loss_file:bool = True,
                       loss_file:str | os.PathLike = 'best_val_loss.pkl'):
        
        if use_loss_file:
            self.loss_file = loss_file

            if os.path.exists(os.path.join(get_calling_file_directory(), loss_file)):
                with open(os.path.join(get_calling_file_directory(), loss_file), 'rb') as file:
                    initial_value_threshold = pickle.load(file)
            else:
                initial_value_threshold = float('inf')

        else:
            self.loss_file = None
            initial_value_threshold = float('inf')
        
        super().__init__(
            filepath = filepath,
            monitor = monitor,
            verbose = verbose,
            save_best_only = save_best_only,
            save_weights_only = save_weights_only,
            mode = mode, 
            save_freq = save_freq,
            initial_value_threshold = initial_value_threshold
        )

    def on_epoch_end(self, epoch, logs=None):
        
        if self.loss_file:
            current_loss = logs.get(self.monitor)
            
            if current_loss is not None and current_loss < self.best:
                with open(os.path.join(get_calling_file_directory(), self.loss_file), 'wb') as file:
                    pickle.dump(current_loss, file)

        super().on_epoch_end(epoch, logs)

