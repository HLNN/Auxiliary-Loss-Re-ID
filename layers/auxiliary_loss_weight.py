import torch
from torch import nn


def flatt_parameter(grad):
    return torch.cat([_.view(-1, 1) for _ in grad], dim=0)


def param_grad_dot(a_grads, b_grads):
    a_grads = flatt_parameter(a_grads)
    b_grads = flatt_parameter(b_grads)
    return torch.matmul(a_grads.view(-1), b_grads.view(-1))


class AuxiliaryWeight(nn.Module):
    def __init__(self, scale=2):
        super(AuxiliaryWeight, self).__init__()
        self.scale = scale
        w = -torch.log(torch.tensor(scale - 1))
        self.weight = nn.Parameter(torch.tensor(w).view(-1), requires_grad=True)

    def forward(self, x=0):
        return self.scale*self.weight.sigmoid()
