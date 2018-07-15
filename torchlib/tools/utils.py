from random import uniform
from urllib.parse import urlparse


HTTP_TIMEOUT = 6.05


def is_url(url):
    return url is not None and urlparse(url).scheme != "" and not os.path.exists(url)

def wait_time():
    """Wait a random number of seconds"""
    return uniform(0.3, 0.5)

