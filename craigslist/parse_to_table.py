from bs4 import BeautifulSoup
from time import strftime, strptime
import requests
import pandas as pd

def parse_timestr(timestr):
    TIME_FORMAT = '%a %d %b %I:%M:%S %p'
    return strptime(timestr, TIME_FORMAT)

class CLPost(object):
    """ Stores informations contained in a Craigslist post
    """
    def __init__(self, orig_soup):
        self.title = None
        self.time_posted = None
        self.content_url = None
        self.listing_price = None
        self.orig_soup = orig_soup

        
    def to_pandas(self):
        return {
            'title': self.title,
            'time_posted': self.time_posted,
            'content_url': self.content_url,
            'listing_price': self.listing_price
        }

        
    def print_post(self):
        print('------------------------------')
        try:
            print(self.title)
        except AttributeError: pass
        try:
            print(strftime('%m/%d %H:%M:%S', self.time_posted))
        except AttributeError: pass
        try:
            print(self.content_url)
        except AttributeError: pass
        try:            
            print('${:.2f}'.format(self.listing_price))
        except AttributeError: pass                    
    

class ParseCraigslist(object):
    @classmethod
    def parse_all_posts(cls, soup):
        rec_list = []
        for post_row in soup.find_all("li", class_="result-row"):
            clpost = cls.parse_post_row(post_row)
            if clpost is not None:
                rec_list.append(clpost.to_pandas())
                clpost.print_post()
                
        df = pd.DataFrame(rec_list)
        print(df)
        df.to_csv('craigslist_posts.df.csv')

    @classmethod
    def parse_post_row(cls, post_row):
        if 'data-pid' not in post_row.attrs:
            return None
        title = post_row.find("a", class_="result-title")        
        if title is None:
            return None

        clpost = CLPost(post_row)
        clpost.title = title.text
        try: # find time posted            
            clpost.time_posted = parse_timestr(post_row.find('time')['title'])
        except NameError: pass

        try: # find url to the actual post page
            content_url = post_row.find('a', class_='result-title')['href']
            if '//' == content_url[:2]:
                content_url = content_url[2:]
            else:
                content_url = 'sfbay.craigslist.org' + content_url
            clpost.content_url = content_url
        except NameError: pass

        try: # find the price
            str_listing_price = post_row.find('span', class_='result-price').text.strip()
            if '$' == str_listing_price[0]:
                listing_price = float(str_listing_price[1:])
                clpost.listing_price = listing_price
        except NameError: pass

        return clpost


if __name__ == "__main__":
    
    params_uniq = {
        'hasPic': 1,
        'search_distance': 14,
        'postal': 94400,
        'pets_cat': 1,
        'bedrooms': 1,
        'bathrooms': 1,
        'min_price': 2400,
        'max_price': 3400,
    }
    haus_loc = 'pen'
    haus_type = 'apa'
    areas = [('nh', nh) for nh in [71, 77, 84, 85, 86, 87, 88]]
    url_params = list(params_uniq.items()) + areas

    req = requests.get('http://sfbay.craigslist.org/search/{}/{}'
                       .format(haus_loc, haus_type), 
                       params=url_params)
    if 200 != req.status_code:
        print('Error: request failed')
        exit(1)
            
    soup = BeautifulSoup(req.content, 'html.parser')
    ParseCraigslist.parse_all_posts(soup)
