import tensorflow as tf

import json
import inspect
import pickle
import os
import pytz

from datetime import datetime, timedelta
from data.Obseration import Observer

from typing import Generator

def get_calling_file_directory():
    
    caller_frame = inspect.stack()[1]
    caller_filename = caller_frame.filename
    
    caller_directory = os.path.dirname(os.path.abspath(caller_filename))
    return caller_directory


class GenerateDataset:

    def __init__(self, alpaca_api_key:str,
                       alpaca_secret_key:str,

                       end: datetime,
                       start: datetime | None = None,

                       news_limit:int = 30,
                       price_limit: int = 50,
                       interval_prices:str = '1h',
                       training:bool = True,
                       use_atr:bool = True,
                       use_rsi:bool = True,
                       use_macd:bool = True,
                       use_obv:bool = True,
                       lenght_atr:int = 14,
                       lenght_rsi:int = 14,
                       lenght_macd: tuple | str = 'standard',
                       neutral:bool = False,

                       settings_filepath:str | None = None,
                       use_first_date:bool = False,
                       first_date_path:str = 'first_date_path.pkl',
                       ):
        
        self.start: datetime = start

        if os.path.exists(os.path.join(get_calling_file_directory(), first_date_path)) and use_first_date:
            with open(os.path.join(get_calling_file_directory(), first_date_path), 'rb') as file:
                self.start = pickle.load(file)

        self.end: datetime = end

        self.date: datetime = self.start
        self.interval_prices = interval_prices

        self.settings_filepath = settings_filepath or 'settings.json'

        self.observer = Observer(
            alpaca_api_key,
            alpaca_secret_key,
            news_limit,
            price_limit,
            interval_prices,
            True,
            use_atr,
            use_rsi,
            use_macd,
            use_obv,
            lenght_atr,
            lenght_rsi,
            lenght_macd,
            neutral
        )

        self.first_date_path = first_date_path


    def update_date(self, symbol:str, date:datetime):

        self.date = self.observer.get_next_date(
            symbol,
            date + timedelta(minutes = 15),
            self.interval_prices
        )

        if self.date.astimezone(pytz.utc) > self.end.astimezone(pytz.utc):
            self.date = self.start


    def save_settings(self, out):

        with open(self.settings_filepath, 'w') as file:
            
            savable = dict()
            
            for key, value in out.items():

                dtype = str(value.dtype)

                if dtype.startswith('int'):
                    dtype = 'int64'
                
                elif dtype.startswith('float'):
                    dtype = 'float32'
                
                else:
                    raise NotImplementedError(f'{dtype} non è supportato')

                savable[key] = {
                    'shape': value.shape,
                    'dtype': dtype
                }

            json.dump(savable, file, indent = 4)


    def get_generator_dataset(self, symbol:str, times:int = 200) -> Generator:

        for _ in range(times):

            self.update_date(symbol, self.date)

            out = self.observer(
                    symbol,
                    self.date
                )

            if not os.path.exists(self.settings_filepath):
                self.save_settings(out)

            yield out
            
    def _create_example(self, data:tf.Tensor, dtype:str):

        data = tf.reshape(data, [-1])

        if dtype.startswith('int'):

            data = tf.cast(data, tf.int64)
            return tf.train.Feature(int64_list = tf.train.Int64List(value = data))
        
        elif dtype.startswith('float'):

            data = tf.cast(data, tf.float32)
            return tf.train.Feature(float_list = tf.train.FloatList(value = data))
        
        else:
            raise NotImplementedError(f'{dtype} non è supportato')
        
            
    def save_dataset_by_num(self, filepath:str | os.PathLike, 
                                  symbol:str, 
                                  times:int = 200,
                                  save_last_date: bool = True):

        generator = self.get_generator_dataset(symbol, times)
        generable = None

        if not os.path.exists(self.settings_filepath):
            out = next(generator)
            self.save_settings(out)

            generable = [out, *generator]

        else:
            generable = generator

            
        with open(self.settings_filepath, 'r') as file:
            settings = json.load(file)

        os.makedirs(os.path.dirname(filepath), exist_ok = True)
           
        with tf.io.TFRecordWriter(filepath) as writer:

            for data in generable:

                feature = {key: self._create_example(value, settings[key]['dtype'])
                       for key, value in data.items()}
                    
                example = tf.train.Example(
                    features = tf.train.Features(
                        feature = feature
                    )
                )
            
                writer.write(example.SerializeToString())

        if save_last_date:

            path = os.path.join(get_calling_file_directory(), self.first_date_path) or \
                os.path.join(get_calling_file_directory(), 'first_date_path.pkl')

            with open(path, 'wb') as file:
                pickle.dump(self.date, file)



class RetrieveDataset:

    def __init__(self, dataset_filepaths:list[str] | list[os.PathLike] | None = None,
                       settings_filepath: str | os.PathLike | None = None):

        if not dataset_filepaths:
            assert os.path.exists('data.tfrecord'), ValueError(f'dataset_filepaths wasn\'t provided')

            dataset_filepaths = ['data.tfrecord']

        if not settings_filepath:
            assert os.path.exists('settings.json'), ValueError(f'settings_filepath wasn\'t provided')

            self.settings_filepath = 'settings.json'

        else:
            self.settings_filepath = settings_filepath

        self.dataset = tf.data.TFRecordDataset(dataset_filepaths)


    def _mapping_func(self):

        with open(self.settings_filepath, 'r') as file:
            settings = json.load(file)

        def _map(proto):

            return tf.io.parse_single_example(
                proto,

                {
                    key: tf.io.FixedLenFeature(shape = item['shape'], dtype = getattr(tf, item['dtype'])) \
                    for key, item in settings.items()
                }
            )

        return _map
    
    def get_size(self):

        counter = 0
        for _ in self.dataset.as_numpy_iterator():
            counter += 1

        return counter

    def get_dataset(self):

        return self.dataset.map(self._mapping_func())