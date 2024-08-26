from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta

class PricesClient:

    def __init__(self, api_key:str, secret_key:str):

        """
        Base client to retrieve the market prices in real time, using the
        alpaca API.

        Inputs: 

        api_key = the alpaca api key
        secret_key = the alpaca api secret key
        """

        self._client = StockHistoricalDataClient(
            api_key, 
            secret_key
        )


    def get_last_price(self, symbol:str, date:datetime, precision:str = '1m'):

        """
        A function to get the last market price before the "date" parameter
        with the "precision" parameter to set the precision/last price round date 
        of the output

        Inputs: \n

        symbol = a market symbol, ex: "AAPL" \n
        date = a datetime object to set the date of the first price \n
        precision = a string set to "number+[h, m, d, M] (hours, minutes, days, months) \n

        Output: \n

        A dataset pandas \n

        The dataset is structured like this: \n

        Index = "timestamp" -> pandas.TimeStamp \n
        Data = ["open", "high", "low", "close", "volume", "trade_count", "vwap"] ->  float64 \n
        """

        return self.get_num_prices(symbol, date, 1, precision)
    

    def get_num_prices(self, symbol:str, date:datetime, num:int, interval:str='1h'):

        """
        A function to get a defined number of market prices, the range starts from
        the datetime input and goes back until it has a sufficient number of
        prices, with "interval" as the difference between the dates.

        Inputs: \n

        symbol = a market symbol, ex: "AAPL" \n
        date = a datetime object to set the date of the first price \n
        num =  the number of prices the function has to return \n
        interval = a string set to "number+[h, m, d, M] (hours, minutes, days, months) \n

        Output: \n

        A dataset pandas \n

        The dataset is structured like this: \n

        Index = "timestamp" -> pandas.TimeStamp \n
        Data = ["open", "high", "low", "close", "volume", "trade_count", "vwap"] ->  float64 \n

        The dataset is guaranteed to be as long as the "num" input specifies
        """

        if interval[-1] == 'h':
            delta = timedelta(days=((int(interval[:-1]) % 24) + 1) * 3)
            
        elif interval[-1] == 'm':
            delta = timedelta(hours=((int(interval[:-1]) % 60) + 1) * 3)

        elif interval[-1] == 'd':
            delta = timedelta(days=int(interval[:-1]) * 7)

        elif interval[-1] == 'M':
            delta = timedelta(days=int(interval[:-1] * 60))

        else:
            raise ValueError(f'Interval: {interval}, not supported, only: (num)s, (num)m, (num)d, (num)M')


        while True:
            
            df = self.get_delta_prices(
                symbol,
                date,
                delta,
                interval
            )

            if len(df) > num:
                break

            else:
                delta += delta

        return df[(len(df) - num):]
    

    def get_data_prices(self, symbol:str, date:datetime, end:datetime, interval:str='1h'):

        """
        A function to get the prices in a range set by (date, end) with 
        "interval" as the difference between the dates, if there are no 
        values in the interval the function will return an empty pandas DataFrame. \n

        Inputs: \n

        symbol = a market symbol, ex: "AAPL" \n
        date = a datetime object to set the date of the first price \n
        end =  a datetime object to set the date of the last price \n
        interval = a string set to "number+[h, m, d, M] (hours, minutes, days, months) \n

        Output: \n

        A dataset pandas with the prices or an empty list if there are no prices in the specified
        range \n

        The dataset is structured like this: \n

        Index = "timestamp" -> pandas.TimeStamp \n
        Data = ["open", "high", "low", "close", "volume", "trade_count", "vwap"] ->  float64 \n
        """

        if interval[-1] == 'h':
            time_frame = TimeFrame(int(interval[:-1]), TimeFrameUnit.Hour)
        
        elif interval[-1] == 'm':
            time_frame = TimeFrame(int(interval[:-1]), TimeFrameUnit.Minute)

        elif interval[-1] == 'd':

            if int(interval[:-1]) == 1:
                time_frame = TimeFrame.Day

            else:
                time_frame = TimeFrame(int(interval[:-1]) * 24, TimeFrameUnit.Hour)

        elif interval[-1] == 'M':
            time_frame = TimeFrame(int(interval[:-1]), TimeFrameUnit.Month)

        else:
            raise ValueError(f'Interval: {interval}, not supported, only: (num)s, (num)m, (num)d, (num)M')

        
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            start = date,
            end = end,
            timeframe = time_frame
        )

        df = self._client.get_stock_bars(request).df

        if len(df.index.names) > 1:
            df_index = df.reset_index()
            df_index = df_index.drop(columns=['symbol'])

            df_new = df_index.set_index('timestamp')

            return df_new
        
        return df
    
    
    def get_delta_prices(self, symbol:str, date:datetime, delta:timedelta, interval:str='1h'):

        '''
        A function to get the prices in a range set by (date - delta, date) with 
        "interval" as the difference between the dates, if there are no 
        values in the interval the function will return an empty pandas DataFrame. \n


        Inputs: \n

        symbol = a market symbol, ex: "AAPL" \n
        date = a datetime object to set the date of the last price \n
        delta = a timedelta object to set the time difference of the last and first price \n
        interval = a string set to "number+[h, m, d, M] (hours, minutes, days, months) \n

        
        Output: \n

        A dataset pandas with the prices or an empty list if there are no prices in the specified
        range \n

        The dataset is structured like this: \n

        Index = "timestamp" -> pandas.TimeStamp \n
        Data = ["open", "high", "low", "close", "volume", "trade_count", "vwap"] ->  float64 \n
        '''

        if interval[-1] == 'h':
            time_frame = TimeFrame(int(interval[:-1]), TimeFrameUnit.Hour)
        
        elif interval[-1] == 'm':
            time_frame = TimeFrame(int(interval[:-1]), TimeFrameUnit.Minute)

        elif interval[-1] == 'd':

            if int(interval[:-1]) == 1:
                time_frame = TimeFrame.Day

            else:
                time_frame = TimeFrame(int(interval[:-1]) * 24, TimeFrameUnit.Hour)

        elif interval[-1] == 'M':
            time_frame = TimeFrame(int(interval[:-1]), TimeFrameUnit.Month)

        else:
            raise ValueError(f'Interval: {interval}, not supported, only: (num)s, (num)m, (num)d, (num)M')

        
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            start = date - delta,
            end = date,
            timeframe = time_frame
        )

        df = self._client.get_stock_bars(request).df

        if len(df.index.names) > 1:
            df_index = df.reset_index()
            df_index = df_index.drop(columns=['symbol'])

            df_new = df_index.set_index('timestamp')

            return df_new
        
        return df