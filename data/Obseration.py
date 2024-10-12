import keras
import tensorflow as tf
import numpy as np
import pandas as pd

from transformers import AutoTokenizer
from data.prices import PricesClient
from data.news import GetNews

from datetime import datetime


class Observer:

    def __init__(self, api_key_alpaca:str,
                       api_secret_alpaca:str,
                       news_limit:int = 30,
                       price_limit:int = 50,
                       interval_prices:str = '1h',
                       training:bool = True,
                       use_atr:bool = True,
                       use_rsi:bool = True,
                       use_macd:bool = True,
                       use_obv:bool = True,
                       lenght_atr:int = 14,
                       lenght_rsi:int = 14,
                       lenght_macd: tuple | str = 'standard',
                       use_neutral:bool = False):

        '''
        Args:

            lenght_macd: It can be "standard", "fast", "slow" or a tuple (short_period, long_period, signal_period). \
                This represent the lenght of the macd.
        '''
        
        supported = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        if interval_prices not in supported:
            raise ValueError(f'Interval dev\'essere in {supported}')

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.client = PricesClient(api_key_alpaca, api_secret_alpaca)
        self.rest = GetNews(api_key_alpaca, api_secret_alpaca)

        self.news_limit = news_limit
        self.price_limit = price_limit
        self.interval_prices = interval_prices

        self.use_atr = use_atr
        self.use_rsi = use_rsi
        self.use_macd = use_macd
        self.use_obv = use_obv
        

        self.lenght_rsi = lenght_rsi
        if not self.use_rsi:
            self.lenght_rsi = 0
        
        self.lenght_atr = lenght_atr
        if not self.use_atr:
            self.lenght_atr = 0

        
        if lenght_macd == 'standard':
            self.lenght_macd = (12, 26, 9)
        
        elif lenght_macd == 'fast':
            self.lenght_macd = (5, 13, 9)

        elif lenght_macd == 'slow':
            self.lenght_macd = (25, 52, 18)

        else:
            self.lenght_macd = lenght_macd

        if not use_macd:
            self.lenght_macd = [0]

        self.training = training
        self.neutral = use_neutral

    
    def get_news_obs(self, symbol:str, date:datetime):                                      

        news = self.rest.get_symbols_by_num(symbol,
                                            self.news_limit,
                                            date,
                                            lambda x: keras.preprocessing.sequence.pad_sequences([self.tokenizer.encode(x)], 50)[0],
                                            lambda x: keras.preprocessing.sequence.pad_sequences([self.tokenizer.encode(''.join(x), truncation = True)], 512)[0]
                                        )[symbol]
        
        return {
            'simbolo': np.array(tf.keras.preprocessing.sequence.pad_sequences(
                                [self.tokenizer.encode(symbol)], 5), dtype=np.int32),

            'titolo': np.array(news['title'], np.int32),

            'paragrafi': np.array(news['paragraphs'], np.int32)
        }
    
    def get_next_date(self, symbol:str, date:datetime, precision:str = '1h') -> datetime:

        return self.client.get_next_price(symbol, date, precision).index[0].to_pydatetime()
    
    
    def get_rsi(self, df:pd.DataFrame, period:int):

        delta = df.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    
    def get_macd(self, df:pd.DataFrame, short_period:int=12, long_period:int=26, signal_period:int=9):

        ema_short = df.ewm(span=short_period, adjust=False).mean()
        ema_long = df.ewm(span=long_period, adjust=False).mean()

        macd = ema_short - ema_long

        signal = macd.ewm(span=signal_period, adjust=False).mean()

        return macd, signal
    

    def get_atr(self, df:pd.DataFrame, period:int):
        
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window = period).mean()

        return atr
    
    
    def get_obv(self, df:pd.DataFrame):

        return pd.Series(np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()


    def __call__(self, symbol:str, date:datetime):

        news_part = self.get_news_obs(symbol, date)

        maximum_len = max(self.lenght_rsi, *self.lenght_macd, self.lenght_atr)

        prices_part = self.client.get_num_prices(
            symbol,
            date,
            self.price_limit + maximum_len,
            self.interval_prices
        )

        if self.use_rsi:
            prices_part['RSI'] = self.get_rsi(prices_part['open'], self.lenght_rsi)
        
        if self.use_macd:
            prices_part['MACD'], prices_part['SIGNAL'] = self.get_macd(prices_part['open'], *self.lenght_macd)
        
        if self.use_atr:
            prices_part['ATR'] = self.get_atr(prices_part, self.lenght_atr)
        
        if self.use_obv:
            prices_part['OBV'] = self.get_obv(prices_part)


        prices_part = prices_part.iloc[maximum_len:]

        if self.training:

            now = self.client.get_next_price(symbol, date, self.interval_prices)['open'].iloc[-1]
            past = prices_part['open'].iloc[-1]

            if now > past:

                label_part = np.array([[1]])

            elif now < past or not self.neutral:
                label_part = np.array([[0]])

            else:
                label_part = np.array([[2]])

            out = {
                'prices': prices_part.values,
                'label': label_part
            }
            out.update(news_part)

            return out
        
        out = {
            'prices': prices_part.values
        }
        out.update(news_part)
        
        return out