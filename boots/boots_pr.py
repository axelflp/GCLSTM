import sys
import os.path
from os.path import exists
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from variables import M

import pickle

from random import choices

########################################################################################################################

with open(f'../data/processed/F/16_6_entrenamiento.pickle', 'rb') as file:
    data_train = pickle.load(file)

print(len(data_train))

with open(f'../data/boots/boots.pickle', 'rb') as file:
    B = pickle.load(file)

print(len(B[0]), B[0][20], len(B[1]), B[1][20], len(B[2]), B[2][20], sep = '\n')















