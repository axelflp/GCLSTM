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

########################################################################################################################
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open(f'../mod_entr/pos0.pickle', 'rb') as file:
    # Call load method to deserialze
    pos = pickle.load(file)

with open(f'../mod_entr/Lap1.pickle', 'rb') as file:
    # Call load method to deserialze
    lap = pickle.load(file)

with open(f'../mod_entr/GCLSTM_ENCODER1.pt', 'rb') as file:
    # Call load method to deserialze
    enc = torch.load(file)

with open(f'../mod_entr/GCLSTM_DECODER1.pt', 'rb') as file:
    # Call load method to deserialze
    dec = torch.load(file)

with open(f'../mod_entr/MLP_OUT1.pt', 'rb') as file:
    # Call load method to deserialze
    ml = torch.load(file)

########################################################################################################################
# DATOS

PATH = f'../data/processed/{HORAS_PASADAS}_{HORAS_POR_PREDECIR}_datos_entrenamiento.pickle'

with open(PATH, 'rb') as file:
    datos = pickle.load(file)

class Info_Dataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            idx = index.tolist()

        return self.dataset[index]


train_data = Info_Dataset(datos)

num_workers = 2
batch_size = 64
dataset_train = DataLoader(train_data, batch_size=batch_size,
                           shuffle=True, num_workers=num_workers,
                           drop_last=True)

########################################################################################################################
# Numero de estaciones
N = 20

Lap_nonze = lap.clone().detach().float().requires_grad_(True)

########################################################################################################################

GCLSTM.Lap_nonze = Lap_nonze
GCLSTM.DEVICE = DEVICE
GCLSTM.pos = pos

GCLSTM_ENCODER = GCLSTM(3, HIDEN_DIMENSIONS, 3, N).to(device=DEVICE)
GCLSTM_DECODER = GCLSTM(3, HIDEN_DIMENSIONS, 3, N).to(device=DEVICE)
MLP_OUT = MLP(HIDEN_DIMENSIONS).to(device=DEVICE)

GCLSTM_ENCODER.load_state_dict(enc, strict=False)
GCLSTM_DECODER.load_state_dict(dec, strict=False)
MLP_OUT.load_state_dict(ml, strict=False)

########################################################################################################################



loss_function = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(list(GCLSTM_ENCODER.parameters()) +
                             list(GCLSTM_DECODER.parameters()) +
                             list(MLP_OUT.parameters()) + [GCLSTM.Lap_nonze],
                             lr=0.0001)

flow = []
epochs = 1000

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

        end = time()
        cont += 1
        print(f'epoca: {i+1}.{cont}, loss: {loss.item()}, tiempo: {end-start}')
        flow.append(loss.item())

torch.save(GCLSTM_DECODER.state_dict(), '../mod_entr/GCLSTM_DECODER1.pt')
torch.save(GCLSTM_ENCODER.state_dict(), '../mod_entr/GCLSTM_ENCODER1.pt')
torch.save(MLP_OUT.state_dict(), '../mod_entr/MLP_OUT1.pt')

pickle.dump(flow, open('../mod_entr/lss1.pkl', 'wb'))

with open('../mod_entr/Lap1.pickle', 'wb') as handle:
    pickle.dump(GCLSTM.Lap_nonze, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../mod_entr/pos0.pickle', 'wb') as handle:
    pickle.dump(GCLSTM.pos, handle, protocol=pickle.HIGHEST_PROTOCOL)
