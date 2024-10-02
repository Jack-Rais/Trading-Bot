import os

import subprocess
import json
import inspect

from collections.abc import Callable
from datetime import datetime, timedelta
from alpaca_trade_api.rest import REST


def get_calling_file_directory():
    
    caller_frame = inspect.stack()[1]
    
    caller_filename = caller_frame.filename
    
    caller_directory = os.path.dirname(os.path.abspath(caller_filename))
    return caller_directory


class GetNews:

    def __init__(self, api_key, secret_key):

        self.rest = REST(api_key, secret_key)

    

    def __preprocess_links_symbols(self, links:list, symbols:list):

        """
        A function to process the links and the symbols before they pass into
        the scraper (that needs to have the links and the symbols in a singol string),
        the function is supposed to be passed directly in the scraper by the termminal.

        Inputs:

        links = a list with the string-links to process
        symbols = a list of lists with the string-symbols to process

        Outputs:

        The links divided by "," (a character that can't be in the link) and 
        the symbols divided by "," for every link and divided by "." for every
        symbol 
        """

        if len(links) != len(symbols):
            raise ValueError('I Simboli devono essere quanti i link')
        
        symbols_out = ''

        for list_symbol in symbols:
            for n, symbol in enumerate(list_symbol):

                symbols_out += symbol

                if n != len(list_symbol) - 1:
                    symbols_out += ','

                else:
                    symbols_out += '.'

        links_out = ','.join(links)

        return links_out, symbols_out

    

    def get_news_by_link(self, links:list, symbols:list):

        """
        Using the Scrapy library the links and the symbols are united to get the title and
        paragraphs of the page, this function uses a json temporary file to put the information
        and then returns them reading the file. \n

        Inputs: \n

        links = a list of the string-links to scrape \n
        symbols = a list of lists of the corresponding symbols of the links \n

        Output: \n
        
        A json like list containing the "title", the "paragraphs" and the "symbols" keys \n
        """

        global_dir = os.getcwd()

        directory_in = get_calling_file_directory()
        directory_out = os.path.abspath(os.path.join(directory_in, '../bezinga'))
        os.chdir(directory_out)

        prep_links, prep_symbols = self.__preprocess_links_symbols(links, symbols)

        subprocess.call(['scrapy', 'crawl', 'news', 
                        '-a', f'start={prep_links}', 
                        '-a', f'symbols={prep_symbols}',
                        '-O', f'{os.path.join(directory_in, "output.json")}:json'], shell=True)
        
        os.chdir(directory_in)

        with open('output.json', 'r', encoding='utf-8') as file:
            out = json.load(file)

        os.chdir(global_dir)

        return out  
    

    

    def get_news_by_date(self, symbol:str | list, 
                               date:datetime, 
                               delta:timedelta, 
                               save_in_file:bool = False, 
                               filename:str = 'output.json', 
                               limit:int = 10, 
                               return_data: bool = True):
          

        news = self.rest.get_news(symbol, 
                                  start=(date - delta).date(), 
                                  end=date.date(),
                                  limit=limit,
                                  exclude_contentless = True)


        links = []
        symbols = []

        for new in news:
            links.append(new.url)
            symbols.append(new.symbols)

        news = self.get_news_by_link(links, symbols)

        if save_in_file:
            global_dir = os.getcwd()
            path = os.path.join(global_dir, filename)

            with open(path, 'w') as file:
                json.dump(news, file, indent=4)

        if return_data:
            return news




    def get_symbols_by_date(self, symbol:str | list[str], 
                                  date:datetime, 
                                  delta:timedelta, 
                                  limit:int = 10,
                                  preprocess_titles:Callable | None = None,
                                  preprocess_paragraphs:Callable | None = None):
        

        if type(preprocess_paragraphs) != type(preprocess_titles):
            raise ValueError('Ci devono essere o due o nessun preprocessor')
        
        symbol = [symbol] if isinstance(symbol, str) else symbol

        news = self.get_news_by_date(symbol, date, delta, limit=limit)

        symbols_news = dict()

        for item in news:

            symbols = item['symbols']
            title = item['title']
            paragraphs = item['paragraphs']

            processed_title = preprocess_titles(title) if preprocess_titles is not None else title
            processed_paragraphs = preprocess_paragraphs(paragraphs) if preprocess_paragraphs is not None \
                                        else paragraphs

            for symbol in symbols:

                if symbol not in symbols_news:

                    symbols_news[symbol] = {
                        'title': [],
                        'paragraphs': []
                    }
                
                symbols_news[symbol]['title'].append(processed_title)
                symbols_news[symbol]['paragraphs'].append(processed_paragraphs)
                   
        return symbols_news
    


    def get_news_by_num(self, symbol:str | list, 
                              nums:int,
                              date:datetime,
                              save_in_file:bool = False, 
                              filename:str = 'output.json', 
                              return_data: bool = True):
    
        delta = timedelta(nums / 5)

        while True:

            news = self.rest.get_news(
                symbol,
                start = (date - delta).date(),
                end = date.date(),
                limit = nums,
                exclude_contentless = True
            )

            if len(news) == nums:
                break

            delta *= 2

        links = []
        symbols = []

        for new in news:
            links.append(new.url)
            symbols.append(new.symbols)

        news = self.get_news_by_link(links, symbols)

        if save_in_file:
            global_dir = os.getcwd()
            path = os.path.join(global_dir, filename)

            with open(path, 'w') as file:
                json.dump(news, file, indent=4)

        if return_data:
            return news


    def get_symbols_by_num(self, symbol:str | list[str], 
                                 nums:int,
                                 date:datetime,
                                 preprocess_titles:Callable | None = None,
                                 preprocess_paragraphs:Callable | None = None):
        
        if type(preprocess_paragraphs) != type(preprocess_titles):
            raise ValueError('Ci devono essere o due o nessun preprocessor')
        
        symbol = [symbol] if isinstance(symbol, str) else symbol

        news = self.get_news_by_num(symbol, nums, date)

        symbols_news = dict()

        for item in news:

            symbols = item['symbols']
            title = item['title']
            paragraphs = item['paragraphs']

            processed_title = preprocess_titles(title) if preprocess_titles is not None else title
            processed_paragraphs = preprocess_paragraphs(paragraphs) if preprocess_paragraphs is not None \
                                        else paragraphs

            for symbol in symbols:

                if symbol not in symbols_news:

                    symbols_news[symbol] = {
                        'title': [],
                        'paragraphs': []
                    }
                
                symbols_news[symbol]['title'].append(processed_title)
                symbols_news[symbol]['paragraphs'].append(processed_paragraphs)
                   
        return symbols_news