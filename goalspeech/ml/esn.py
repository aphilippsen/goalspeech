import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class ESN(nn.Module):
    """
        Echo state network
    """

    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh', density=0.2, leak_rate=1, spectral_radius=0.95, lambda_reg = 1, weight_seed=None):
        super(ESN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lambda_reg = lambda_reg

        self.reservoir = Reservoir(input_dim, hidden_dim, nonlinearity=nonlinearity, density=density, leak_rate=leak_rate, spectral_radius=spectral_radius, seed=weight_seed)

        self.readout = nn.Linear(hidden_dim, output_dim, bias=False)

        self.HTH = None
        self.HTY = None

    # feed input data into the network and generate the necessary matrices for the readout calculations
    def forward(self, input, target, washout=10):
        with torch.no_grad():
            hidden, activations = self.reservoir(input)

            if target.size()[0] > 4*washout:
                activations = activations[washout:,:]
                target = target[washout:,:]
            else:
                print("Adjust washout of ESN to a smaller number!")

            # output = self.readout(activations)
            self.HTH = torch.mm(activations.t(), activations)
            self.HTY = torch.mm(activations.t(), target)

    # calculate readout weights via pseudoinverse
    def calculate_readout(self):
        with torch.no_grad():
            I = (self.lambda_reg * torch.eye(self.HTH.size(0))).to(self.HTH.device)
            A = self.HTH + I

            if torch.det(A) != 0:
                W = torch.mm(torch.inverse(A), self.HTY).t()
            else:
                pinv = torch.pinverse(A)
                W = torch.mm(pinv, self.HTY).t()

            self.readout.weight = nn.Parameter(W.contiguous())

        self.HTH = None
        self.HTY = None
        self.reservoir.reset()


class Reservoir(nn.Module):
    def __init__(self, input_dim, hidden_dim, leak_rate = 1, nonlinearity='tanh', density=0.2, spectral_radius=0.95, w_ih_scale=1, seed=None):
        super(Reservoir, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.nonlinearity = nonlinearity
        self.leak_rate = leak_rate
        self.density = density
        self.spectral_radius = spectral_radius

        # create random weights according to seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # input weights
        self.w_ih = nn.Parameter(torch.Tensor(self.hidden_dim, self.input_dim))
        self.b_ih = nn.Parameter(torch.Tensor(self.hidden_dim))
        nn.init.uniform_(self.w_ih, -1, 1)
        self.w_ih.data *= w_ih_scale
        nn.init.uniform_(self.b_ih, -1, 1)
        self.b_ih.data *= w_ih_scale

        # reservoir weights
        self.w_hh = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))

        w_hh = torch.Tensor(self.hidden_dim * self.hidden_dim)
        w_hh.uniform_(-1, 1)
        if density < 1:
            zero_weights = torch.randperm(int(self.hidden_dim * self.hidden_dim))
            zero_weights = zero_weights[:int(self.hidden_dim * self.hidden_dim * (1 - density))]
            w_hh[zero_weights] = 0
        w_hh = w_hh.view(self.hidden_dim, self.hidden_dim)
        abs_eigs = (torch.eig(w_hh)[0] ** 2).sum(1).sqrt()
        self.w_hh.data = w_hh * (self.spectral_radius / torch.max(abs_eigs))

        # initial activations
        self.reset()

        torch.initial_seed()
        np.random.seed(None)

    def reset(self):
        self.activations = torch.zeros(self.hidden_dim)
        self.hidden_state = torch.zeros(self.hidden_dim)

    def forward(self, input):

        all_activations = torch.zeros((input.shape[0], self.hidden_dim))
        all_hidden = torch.zeros((input.shape[0], self.hidden_dim))

        for i in range(input.shape[0]):
            next_hidden_state = F.linear(input[i,:], self.w_ih, self.b_ih) + F.linear(self.activations, self.w_hh)
            next_hidden_state = (1-self.leak_rate) * self.hidden_state + self.leak_rate * next_hidden_state

            if self.nonlinearity == 'tanh':
                next_act = torch.tanh(next_hidden_state)
            elif self.nonlinearity == 'relu':
                next_act = F.relu(next_hidden_state)

            all_activations[i,:] = next_act
            all_hidden[i,:] = next_hidden_state

            self.activations = next_act
            self.hidden_state = next_hidden_state

        return all_activations, all_hidden

