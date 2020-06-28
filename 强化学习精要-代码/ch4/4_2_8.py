# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:22:14 2020

@author: dell
"""

from contextlib import contextmanager
import time

@contextmanager
def timer(prefix):
    start_time = time.time()
    yield()
    duration = time.time() - start_time
    print( prefix + str(duration))


if __name__ == "__main__":
    with timer ('corning') :
        # my code
        print("my_code:  hello, corning")
        time.sleep(3)
