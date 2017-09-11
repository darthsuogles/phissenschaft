"""
Login and search

http://blog.just4fun.site/LinkedIn-spider-note.html
"""

import requests
from bs4 import BeautifulSoup

sess = requests.Session()
ROOT_URL = 'https://www.linkedin.com'
LOGIN_URL = '{}/uas/login-submit'.format(ROOT_URL)
SEARCH_URL = '{}/search/results/index/?keywords=machine%20learning&origin=GLOBAL_SEARCH_HEADER'.format(ROOT_URL)

html = sess.get(ROOT_URL).content
soup = BeautifulSoup(html)
csrf = soup.find(id='loginCsrfParam-login')['value']

login_info = {
    'session_key': '<YOUR_ACCOUNT>',
    'session_password': '<YOUR_PASSWORD>',
    'loginCsrfParam': csrf
}

sess.post(LOGIN_URL, data=login_info)
resp = sess.get(SEARCH_URL)
print(resp, 'should be', 200)

""" Get selenum drive
"""
from selenium import webdriver
driver = webdriver.Chrome()

driver.get(SEARCH_URL)
print(driver.title)
