import scrapy
from scrapy.http.request import Request
from scrapy.http.response import Response
from ..items import BezingaItem


class NewsSpider(scrapy.Spider):
    name = "news"
    allowed_domains = ["www.benzinga.com"]
    start_urls = ["https://www.benzinga.com/"]

    user_agents = [
        'Mozilla/5.0 (iPhone; CPU iPhone OS 8_5_9; like Mac OS X) AppleWebKit/536.27 (KHTML, like Gecko)  Chrome/48.0.1640.204 Mobile Safari/601.0',
        'Mozilla/5.0 (Linux x86_64; en-US) AppleWebKit/533.6 (KHTML, like Gecko) Chrome/50.0.2399.392 Safari/602',
        'Mozilla/5.0 (Linux; Android 5.1.1; SAMSUNG SM-G935S Build/MMB29M) AppleWebKit/602.6 (KHTML, like Gecko)  Chrome/47.0.3571.298 Mobile Safari/534.2',
        'Mozilla/5.0 (Windows; U; Windows NT 6.1; Win64; x64) AppleWebKit/534.8 (KHTML, like Gecko) Chrome/47.0.2094.129 Safari/536',
        'Mozilla/5.0 (Windows; U; Windows NT 6.3; Win64; x64; en-US) AppleWebKit/533.18 (KHTML, like Gecko) Chrome/51.0.1228.163 Safari/535'
    ]

    def __init__(self, start=None, symbols=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(start, list):
            self.start_urls = start
            self.symbols = symbols

        elif isinstance(start, str):
            self.start_urls = start.replace(' ', '').split(',')
            
            self.symbols = symbols.replace(' ', '').split('.')
            symbols = []

            for x in self.symbols:
                symbols.append(x.split(','))

            self.symbols = symbols
        
        else:
            raise ValueError('Input deve contenere dei link')
        

    def start_requests(self):
        for i, url in enumerate(self.start_urls):
            user_agent = self.user_agents[i % len(self.user_agents)]
            symbol = self.symbols[i]
            yield Request(url, 
                          headers={'User-Agent': user_agent}, 
                          meta={'symbol': symbol}, 
                          callback=self.parse)

    def togli(self, stringa):
        found = False
        result = ''
        for x in stringa:
            if x == '<':
                found = True
            elif x == '>':
                found = False
            else:
                if not found:
                    result += x
        return result

    def parse(self, response: Response):
        
        title = response.xpath('//div/div/h1/text()').get()
        paragraphs = response.xpath('//div[@id="article-body"]/div/p').getall()
        symbol = response.meta['symbol']

        text = []

        for x in paragraphs:
            text.append(self.togli(x))

        books = BezingaItem()

        books['title'] = title
        books['symbols'] = symbol
        books['paragraphs'] = text

        yield books
