#encoding=utf-8
import sys
import os
current_url = os.path.dirname(__file__)
parent_url = os.path.abspath(os.path.join(current_url, os.pardir))

sys.path.append(parent_url)

from data_utils import load_dataset