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

R = len(data_train)
B = []

for i in range(M):
    b_a = choices(data_train, k=R)
    B.append(b_a)

with open(f'../data/boots/boots.pickle', 'wb') as handle:
    pickle.dump(B, handle, protocol=pickle.HIGHEST_PROTOCOL)















