import json
import pickle

from pathlib import Path
from pos_tagger.memm import MEMM
from utils.parser import Parser


with open(r"models\05-27_00-27-51\trained_weights_data_1.pkl", 'rb') as f:
    weights = pickle.load(f)

print("a")