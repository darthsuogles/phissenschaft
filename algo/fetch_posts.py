
import requests
import regex
import json
from bs4 import BeautifulSoup
from pathlib import Path

def get_summary_page(params):
    res = requests.get(
        r"http://hr.tencent.com/position.php", 
        params=params
    )
    pg_elems = BeautifulSoup(res.text, 'html.parser')
    return {'url': res.url, 'elems': pg_elems}


def get_summary_pages(params):
    # Find all paginated pages
    init_page = get_summary_page(params)
    summary_pages = [init_page]

    summary_paginate_patt = regex.compile(r'position\.php\?.*start=(\d+)')
    
    max_iters = 7
    for curr_iter in range(max_iters):
        start_inds = []
        for _page_url_res in init_page['elems'].find_all(class_="pagenav"):
            for _pg in _page_url_res('a', href=summary_paginate_patt): 
                _inds = summary_paginate_patt.findall(_pg.attrs['href'])
                _idx = list(map(int, _inds))[0]
                start_inds.append(_idx)

        if not start_inds:
            continue
        start_inds = sorted(list(set(start_inds)))

    for pg_idx in start_inds: 
        _params = params.copy()
        _params.update({'start': pg_idx})
        print("working on paginated summary page", pg_idx)
        _res = get_summary_page(_params)
        summary_pages.append(_res)

    return summary_pages


# Get the detail page of a specific post
def get_post_details(post_attrs):
    post_id, post_name = post_attrs['id'], post_attrs['name']

    base_url = "http://hr.tencent.com"
    post_detail_url = r"{}/position_detail.php".format(base_url)
    try:
        _res = requests.get(post_detail_url, params={'id': post_id})
        post_attrs.update({'url': _res.url})
    except:
        print('failed to obtain result: {}'.format(_res.url))
        return post_attrs

    _soup = BeautifulSoup(_res.text, 'html.parser')
    try:
        _pg_elems = _soup.find_all(id='position_detail')[0].find_all('td')
    except:
        print('failed to find relevant elemnets in parsed HTML page')
        return post_attrs
            
    for _pg_elem in _pg_elems:
        details = _pg_elem.find_all(class_='squareli')
        if not details: 
            continue
        _typ = _pg_elem.text.split("：")[0].strip()
        _parsed = {_typ: list(map(lambda pg: pg.text, _pg_elem('li')))}
        post_attrs.update(_parsed)

    return post_attrs


# Given a result page, find all post details
def get_init_post_details(summary_pages):
    post_detail_attr_list = []
    post_detail_patt = regex.compile(r'position_detail\.php\?id=(\d+)')
    for summary_page in summary_pages:
        for _pg in summary_page['elems'].find_all(href=post_detail_patt):
            _m = post_detail_patt.findall(_pg.attrs['href'])
            _idx = list(map(int, _m))[0]
            post_detail_attr_list.append({'id': _idx, 'name': _pg.text})

    return post_detail_attr_list


def find_all_posts(keywords):
    """ Find all posts with matching keywords
    """
    summary_params = {
        'keywords': keywords,
        'tid': 0,  # or 87 for technical
    }
    summary_pages = get_summary_pages(summary_params)    
    post_detail_attr_list = get_init_post_details(summary_pages)

    base_attr_types = set(['id', 'name', 'url'])
    max_iters = 11

    for curr_iter in range(max_iters):
        is_changed = False
        for i, _attr in enumerate(post_detail_attr_list):
            if base_attr_types < _attr.keys():
                continue  
            is_changed = True
            full_attr = get_post_details(_attr)
            print(full_attr)
            post_detail_attr_list[i] = full_attr

        if not is_changed:
            break

    if curr_iter == max_iters and is_changed:
        print('Warning: we did not get all the post details')

    return post_detail_attr_list


fp_rel_posts = Path('tencent_relevant_posts.json')

try:
    uniq_rel_posts = json.load(fp_rel_posts.open('r'))
except:
    posts_ai = find_all_posts("人工智能")
    posts_ml = find_all_posts("机器学习")
    posts_dl = find_all_posts("深度学习")

    all_rel_posts = posts_ai + posts_ml + posts_dl
    uniq_rel_posts = {}
    for post_d in all_rel_posts:
        try:
            _od = uniq_rel_posts[post_d['id']]
            assert(_od == post_d)
        except:
            uniq_rel_posts[post_d['id']] = post_d

    json.dump(uniq_rel_posts, 
              fp_rel_posts.open('w'), 
              ensure_ascii=False)

# Find stuffs
import pandas as pd
df = pd.DataFrame.from_dict(uniq_rel_posts, orient='index')

def has_keywords(words, op):
    words = [w.lower() for w in words]
    def fn_(sents):
        text = ' '.join(sents).lower()
        return op([w in text for w in words])
    return fn_    

def has_any(words):
    return has_keywords(words, any)
    
def has_all(words):
    return has_keywords(words, all)

_inds = df['工作职责'].apply(has_any(['linux']))
_inds_mig = df['name'].apply(lambda nm: nm.startswith('MIG'))

df[_inds]

print(df.ix[27157].url)

df.to_csv('tencent_relevant_posts.df.csv')

# _inds_mig_sel = [
#     25765,
#     26188,
#     26422,
#     27517,
#     27637,
#     27686,
#     27842
# ]

# df.ix[_inds_mig_sel]
