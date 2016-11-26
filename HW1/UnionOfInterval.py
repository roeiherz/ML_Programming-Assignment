import numpy as np
import datetime
from sklearn.datasets import fetch_mldata
from PIL import Image
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

__author__ = 'roeiherz & mosheraboh'

if __name__ == '__main__':
    print 'start'
    # Start time
    start_time = datetime.datetime.now()
    print 'the start is at {}'.format(start_time)

    # End time
    end_time = datetime.datetime.now()
    time_diff = end_time - start_time
    print 'this algo runs in {0} '.format(time_diff)
    print 'end'