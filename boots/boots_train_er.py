#!/usr/bin/env python3

import sys
import os.path
from os.path import exists
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# Torch functions
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# Manipulation functions
import pandas as pd
import numpy as np

import random

import pickle

from variables import HORAS_PASADAS, HORAS_POR_PREDECIR, HIDEN_DIMENSIONS
from models import GCLSTM, MLP

from time import time



############################################################################################################

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATH = f'../data/boots/boots.pickle'

file_exists = exists(PATH)

if not file_exists:
    import boots

with open(PATH, 'rb') as file:
    B = pickle.load(file)


class Info_Dataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            idx = idx.tolist()

        return self.dataset[index]


#####################################################################
with open('../data/processed/matriz_ady.pickle', 'rb') as file:
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

###############################################################

GCLSTM.DEVICE = DEVICE
GCLSTM.pos = pos

flow = []
epochs = 100
ep_sav = [49, 99, 149,100]

###############################################################################

B = B[:10]
for s, b_s in enumerate(B):

    train_data = Info_Dataset(b_s)

    num_workers = 2
    batch_size = 64
    dataset_train = DataLoader(train_data, batch_size=batch_size,
                               shuffle=True, num_workers=num_workers,
                               drop_last=True)

    GCLSTM.Lap_nonze = torch.tensor(nonze).float().requires_grad_()

    GCLSTM_E = GCLSTM(3, HIDEN_DIMENSIONS, 3, N).to(device=DEVICE)
    GCLSTM_D = GCLSTM(3, HIDEN_DIMENSIONS, 3, N).to(device=DEVICE)
    MLP_OUT = MLP(HIDEN_DIMENSIONS).to(device=DEVICE)

    loss_function = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(list(GCLSTM_E.parameters()) +
                                 list(GCLSTM_D.parameters()) +
                                 list(MLP_OUT.parameters()) + [GCLSTM.Lap_nonze],
                                 lr=0.0001)

    flow_aux = []
    start = time()

    for i in range(epochs):
        cont = 0

        for j in dataset_train:

            reales = j['p'].to(device=DEVICE)
            pred_bat = torch.zeros(batch_size, N, HORAS_POR_PREDECIR, device=DEVICE)

            for l in range(batch_size):

                h = torch.zeros(N, HIDEN_DIMENSIONS, device=DEVICE).float()
                c = torch.zeros(N, HIDEN_DIMENSIONS, device=DEVICE).float()

                encoder = j['encoder'][l].to(device=DEVICE)
                decoder = j['decoder'][l].to(device=DEVICE)

                # encoder
                for k in encoder:
                    h, c = GCLSTM_E(k, h, c)
                # decoder
                for k in decoder:
                    h, c = GCLSTM_D(k, h, c)
                # mlp
                pred_bat[l, :, :] = MLP_OUT(h)

            # descenso
            optimizer.zero_grad()
            loss = loss_function(pred_bat, reales)
            loss.backward()
            optimizer.step()

            end = time()
            cont += 1
            print(f'B{s} --  epoca: {i + 1}.{cont}, loss: {loss.item()}, tiempo: {end - start}')
            flow_aux.append(loss.item())

        if i in ep_sav:

            torch.save(GCLSTM_D.state_dict(), f'../mod_entr/boots/GCLSTM_DECODER_B{s}_{i+1}_ep.pt')
            torch.save(GCLSTM_E.state_dict(), f'../mod_entr/boots/GCLSTM_ENCODER_B{s}_{i+1}_ep.pt')
            torch.save(MLP_OUT.state_dict(), f'../mod_entr/boots/MLP_OUT_B{s}_{i+1}_ep.pt')

            with open(f'../mod_entr/boots/Laplaciano_B{s}_{i+1}_ep.pickle', 'wb') as handle:
                pickle.dump(GCLSTM.Lap_nonze, handle, protocol=pickle.HIGHEST_PROTOCOL)

        flow.append(flow_aux)

