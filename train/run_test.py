import sys
import os.path

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# Torch functions
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# Manipulation functions
import pandas as pd
import numpy as np

from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

import pickle

from variables import HORAS_PASADAS, HORAS_POR_PREDECIR, HIDEN_DIMENSIONS
from models.GCLSTM_modelo import GCLSTM
from models.MLP_modelo import MLP

from time import time
import multiprocessing as mp

########################################################################################################################
def set_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def set_seed(seed=None, seed_torch=True):
    if seed is None:
        seed = np.random.choice(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


DEVICE = set_device()
set_seed(755)
print(DEVICE)
########################################################################################################################
# DATOS
print(GCLSTM.__mro__, MLP.__mro__, sep='\n')
PATH = f'processed/{HORAS_PASADAS}_{HORAS_POR_PREDECIR}_datos_procesados.pickle'

#dat = input(f'Nombre de los datos procesados (Sin extension) [default: {HORAS_PASADAS}_{HORAS_POR_PREDECIR}_datos_procesados.pickle]:')

#if dat != '':
#    PATH = f'processed/{dat}.pickle'


with open(f'../data/{PATH}', 'rb') as file:
    # Call load method to deserialze
    datos_procesados = pickle.load(file)

# Dividimos en datos de entrenamiento y datos para testear
_, data_ts = train_test_split(datos_procesados, test_size=0.005, random_state=42)

print(len(data_ts))

class Info_Dataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            idx = idx.tolist()

        return self.dataset[index]


test_data = Info_Dataset(data_ts)


########################################################################################################################
# LAPLACIANO

with open('../data/processed/matriz_ady.pickle', 'rb') as file:
    # Call load method to deserialze
    mat_adj = pickle.load(file)

adjacent_matrix = mat_adj['Matriz de adyacencia']

# Creamos la matriz diagonal D
D = np.diag(np.sum(adjacent_matrix, axis=0))

# Creamos el Laplaciano
L = D - adjacent_matrix

# Normalizamos
eigva, _ = np.linalg.eig(L)
max_eig = np.max(eigva)
L = 2 * L / max_eig - np.identity(adjacent_matrix.shape[0])

# Numero de estaciones
N = L.shape[0]

nonze = []
pos = []
for i in range(L.shape[0]):
    for j in range(L.shape[1]):
        if L[i][j] != 0:
            nonze.append(L[i][j])
            pos.append((i, j))

Lap_nonze = torch.tensor(nonze).float().requires_grad_()
#l1 = Lap_nonze.clone().detach()

########################################################################################################################

GCLSTM.Lap_nonze = Lap_nonze
#GCLSTM.Lap_nonze.float().requires_grad_()
GCLSTM.DEVICE = DEVICE
GCLSTM.pos = pos

GCLSTM_ENCODER = GCLSTM(3, HIDEN_DIMENSIONS, 3, N).to(device=DEVICE)
GCLSTM_DECODER = GCLSTM(3, HIDEN_DIMENSIONS, 3, N).to(device=DEVICE)
MLP_OUT = MLP(HIDEN_DIMENSIONS).to(device=DEVICE)

########################################################################################################################
print(GCLSTM.Lap_nonze)

loss_function = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(list(GCLSTM_ENCODER.parameters()) +
                             list(GCLSTM_DECODER.parameters()) +
                             list(MLP_OUT.parameters()) + [GCLSTM.Lap_nonze],
                             lr=5)
flow = []
epochs = 3 

#print(GCLSTM.Lap_nonze)


for batch_size in [64]:
    for num_workers in [2]:
        dataset_test = DataLoader(test_data, batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers, drop_last=True)

        start = time()

        for i in range(epochs):

            for j in dataset_test:

                #GCLSTM.Lap_nonze = torch.clone(Lap_nonze) 

                reales = j['p'].to(device=DEVICE)
                pred_bat = torch.zeros(batch_size, N, HORAS_POR_PREDECIR, device=DEVICE)

                for l in range(batch_size):

                    h = torch.zeros(N, HIDEN_DIMENSIONS, device=DEVICE).float()
                    c = torch.zeros(N, HIDEN_DIMENSIONS, device=DEVICE).float()

                    encoder = j['encoder'][l].to(device=DEVICE)
                    decoder = j['decoder'][l].to(device=DEVICE)

                    # encoder
                    for k in encoder:
                        h, c = GCLSTM_ENCODER(k, h, c)
                    # decoder
                    for k in decoder:
                        h, c = GCLSTM_DECODER(k, h, c)
                    # mlp
                    pred_bat[l, :, :] = MLP_OUT(h)

                # descenso
                optimizer.zero_grad()
                loss = loss_function(pred_bat, reales)
                loss.backward()
                optimizer.step()
                

                flow.append(loss.item())
                print(GCLSTM.Lap_nonze)
                #print(list(GCLSTM_ENCODER.parameters()))

        end = time()
        print("Finish with:{} second, batch size={}, num_workers={}".format(end - start, batch_size, num_workers))
        #print(GCLSTM.Lap_nonze)
#        l2 = GCLSTM.Lap_nonze.clone().detach()
#        print(torch.all(l1.eq(l2)))
#
# pickle.dump(GCLSTM_DECODER, open('trained_model/GCLSTM_D.pkl', 'wb'))
# pickle.dump(GCLSTM_ENCODER, open('trained_model/GCLSTM_E.pkl', 'wb'))
# pickle.dump(MLP_OUT, open('trained_model/MLP.p
