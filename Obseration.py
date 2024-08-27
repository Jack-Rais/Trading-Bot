import tensorflow as tf
import numpy as np

from transformers import AutoTokenizer
from prices import PricesClient
from news import GetNews

from datetime import datetime, timedelta


class Observer:

    def __init__(self, api_key_alpaca:str,
                       api_secret_alpaca:str,
                       news_limit:int = 30,
                       price_limit:int = 50,
                       interval_prices:str = '1h',
                       training:bool = True,
                       use_neutral:bool = False):
        
        supported = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        if interval_prices not in supported:
            raise ValueError(f'Interval dev\'essere in {supported}')

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.client = PricesClient(api_key_alpaca, api_secret_alpaca)
        self.rest = GetNews(api_key_alpaca, api_secret_alpaca)

        self.news_limit = news_limit
        self.price_limit = price_limit
        self.interval_prices = interval_prices

        self.training = training
        self.neutral = use_neutral

    
    def get_news_obs(self, symbol:str, date:datetime):

        def titles(x):

            return tf.keras.preprocessing.sequence.pad_sequences(
                            self.tokenizer(x, return_tensors='tf')['input_ids'],
                            maxlen=100)[0]
        
        def paragraphs(x):

            if x == []:

                return tf.keras.preprocessing.sequence.pad_sequences(
                            self.tokenizer([self.tokenizer.pad_token],
                                return_tensors='tf')['input_ids'],
                            maxlen=512)[0] 
            
            else:

                return tf.keras.preprocessing.sequence.pad_sequences(
                            self.tokenizer(x, 
                                padding=True, 
                                truncation=True,
                                return_tensors='tf')['input_ids'],
                            maxlen=512)[0] 
                                        

        news = self.rest.get_symbols_by_num(symbol,
                                            self.news_limit,
                                            date,
                                            titles, 
                                            paragraphs
                                        )[symbol]
        
        return {
            'simbolo': np.array(tf.keras.preprocessing.sequence.pad_sequences(
                            self.tokenizer(symbol, return_tensors='tf')['input_ids'],
                            maxlen=5), dtype=np.int32),

            'titolo': np.array(news['title'], np.int32),

            'paragrafi': np.array(news['paragraphs'], np.int32)
        }


    def __call__(self, symbol:str, date:datetime):

        news_part = self.get_news_obs(symbol, date)
        prices_part = self.client.get_num_prices(symbol, date, self.price_limit, self.interval_prices)

        if self.training:

            now = self.client.get_next_price(symbol, date, self.interval_prices)['open'][-1]
            past = prices_part['open'][-1]

            if now > past:

                label_part = np.array([[1]])

            elif now < past or not self.neutral:
                label_part = np.array([[0]])

            else:
                label_part = np.array([[2]])

            return {
                'news': news_part,
                'prices': prices_part,
                'label': label_part
            }
        
        return {
                'news': news_part,
                'prices': prices_part
        }