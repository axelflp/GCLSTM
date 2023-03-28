import torch
import torch.nn as nn

# Manipulation functions
import numpy as np


class GCLSTM(nn.Module):

    Lap_nonze = None
    pos = None
    DEVICE = "cpu"

    def __init__(self, input_size, hidden_size, k=4, N=0):
        super(GCLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        self.N = N
        # Definimos los parámetros que usaremos para el encoder
        # f(t)

        self.W_fh = nn.Parameter(
            torch.zeros(self.k, self.hidden_size, self.hidden_size).uniform_(-np.sqrt(1. / self.input_size),
                                                                             np.sqrt(1. / self.input_size)))
        self.W_fx = nn.Parameter(
            torch.zeros(self.k, self.hidden_size, self.input_size).uniform_(-np.sqrt(1. / self.input_size),
                                                                            np.sqrt(1. / self.input_size)))
        self.bf = nn.Parameter(torch.zeros(self.N, self.hidden_size).uniform_(-np.sqrt(1. / self.input_size),
                                                                              np.sqrt(1. / self.input_size)))
        # i(t)
        self.W_ih = nn.Parameter(
            torch.zeros(self.k, self.hidden_size, self.hidden_size).uniform_(-np.sqrt(1. / self.input_size),
                                                                             np.sqrt(1. / self.input_size)))
        self.W_ix = nn.Parameter(
            torch.zeros(self.k, self.hidden_size, self.input_size).uniform_(-np.sqrt(1. / self.input_size),
                                                                            np.sqrt(1. / self.input_size)))
        self.bi = nn.Parameter(torch.zeros(self.N, self.hidden_size).uniform_(-np.sqrt(1. / self.input_size),
                                                                              np.sqrt(1. / self.input_size)))

        # o(t)
        self.W_oh = nn.Parameter(
            torch.zeros(self.k, self.hidden_size, self.hidden_size).uniform_(-np.sqrt(1. / self.input_size),
                                                                             np.sqrt(1. / self.input_size)))
        self.W_ox = nn.Parameter(
            torch.zeros(self.k, self.hidden_size, self.input_size).uniform_(-np.sqrt(1. / self.input_size),
                                                                            np.sqrt(1. / self.input_size)))
        self.bo = nn.Parameter(torch.zeros(self.N, self.hidden_size).uniform_(-np.sqrt(1. / self.input_size),
                                                                              np.sqrt(1. / self.input_size)))

        # c(t)
        self.W_ch = nn.Parameter(
            torch.zeros(self.k, self.hidden_size, self.hidden_size).uniform_(-np.sqrt(1. / self.input_size),
                                                                             np.sqrt(1. / self.input_size)))
        self.W_cx = nn.Parameter(
            torch.zeros(self.k, self.hidden_size, self.input_size).uniform_(-np.sqrt(1. / self.input_size),
                                                                            np.sqrt(1. / self.input_size)))
        self.bc = nn.Parameter(torch.zeros(self.N, self.hidden_size).uniform_(-np.sqrt(1. / self.input_size),
                                                                              np.sqrt(1. / self.input_size)))

    def cheb_pol_mat(self, K, mat):
        if K == 0:
            return torch.matrix_power(mat, 0)
        else:
            if K == 1:
                return mat
            else:
                return 2 * mat @ self.cheb_pol_mat(K - 1, mat) - self.cheb_pol_mat(K - 2, mat)


    def graph_conv(self, X, W, L):
        conv = torch.zeros(L.shape[0], W.shape[1], device=GCLSTM.DEVICE).float()
        for i in range(0, W.shape[0]):
            conv_aux = self.cheb_pol_mat(i, L)
            conv_aux = torch.matmul(conv_aux, X)
            conv_aux = torch.matmul(conv_aux, W[i, :, :].T)
            conv = conv + conv_aux
        return conv

    def forward(self, x, h, c):
        Lap = torch.zeros(x.shape[0], x.shape[0], device=GCLSTM.DEVICE).float()
        for n, idx in enumerate(GCLSTM.pos):
            Lap[idx[0]][idx[1]] = GCLSTM.Lap_nonze[n]
#        Lap.requires_grad_()
        f = torch.sigmoid(self.graph_conv(h, self.W_fh, Lap) + self.graph_conv(x, self.W_fx, Lap) + self.bf)
        i = torch.sigmoid(self.graph_conv(h, self.W_ih, Lap) + self.graph_conv(x, self.W_ix, Lap) + self.bi)
        o = torch.sigmoid(self.graph_conv(h, self.W_oh, Lap) + self.graph_conv(x, self.W_ox, Lap) + self.bo)
        c = i * torch.tanh(self.graph_conv(h, self.W_ch, Lap) + self.graph_conv(x, self.W_cx, Lap) + self.bc) + f * c
        h = o * torch.tanh(c)

        return h, c
