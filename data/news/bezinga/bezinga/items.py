# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class BezingaItem(scrapy.Item):
    title = scrapy.Field()
    symbols = scrapy.Field()
    paragraphs = scrapy.Field()
