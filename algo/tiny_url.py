"""
Desing a TinyURL system
"""
from base64 import b64encode, b64decode
import hashlib

url = "https://docs.python.org/3/library/base64.html"
m = hashlib.md5()
m.update(bytes(url, 'ascii'))
# Originally 62 -> +, 63 -> /
# We must use alternative characters to form valid path names
# https://docs.python.org/3/library/base64.html#base64.urlsafe_b64decode
encoded = b64encode(m.digest(), b'-_')
