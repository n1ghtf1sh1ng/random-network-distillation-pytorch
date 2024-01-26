#
# Dynamic Routing Between Capsules
# https://arxiv.org/pdf/1710.09829.pdf
#

import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def squash(s, dim):
  # This is Eq.1 from the paper.
  mag_sq = torch.sum(s**2, dim=dim, keepdim=True)
  mag = torch.sqrt(mag_sq)
  s = (mag_sq / (1.0 + mag_sq)) * (s / mag)

  return s


class Conv1(nn.Module):
  def __init__(self):
    super(Conv1, self).__init__()

    self.conv = nn.Conv2d(
      in_channels=1,
      out_channels=16,
      kernel_size=3,
      stride=4,
      bias=True
    )

    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    # x: [batch_size, 1, 84, 84]

    h = self.relu(self.conv(x))
    # h: [batch_size, 16, 21, 21]

    return h


class ConvUnit(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ConvUnit, self).__init__()

    self.conv = nn.Conv2d(
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=9,
      stride=2,
      bias=True
    )

  def forward(self, x):
    # x: [batch_size, in_channels=16, 21, 21]

    h = self.conv(x)
    # h: [batch_size, out_channels=8, 7, 7]

    return h


class PrimaryCaps(nn.Module):
  def __init__(self):
    super(PrimaryCaps, self).__init__()

    self.conv1_out = 16 # out_channels of Conv1, a ConvLayer just before PrimaryCaps
    self.capsule_units = 49
    self.capsule_size = 8

    def create_conv_unit(unit_idx):
        unit = ConvUnit(
          in_channels=self.conv1_out,
          out_channels=self.capsule_size
        )
        self.add_module("unit_" + str(unit_idx), unit)
        return unit

    self.conv_units = [create_conv_unit(i) for i in range(self.capsule_units)]

  def forward(self, x):
    # x: [batch_size, 16, 21, 21]
    batch_size = x.size(0)

    u = []
    for i in range(self.capsule_units):
      u_i = self.conv_units[i](x)
      # u_i: [batch_size, capsule_size=8, 7, 7]

      u_i = u_i.view(batch_size, self.capsule_size, -1, 1)
      # u_i: [batch_size, capsule_size=8, 49, 1]

      u.append(u_i)
    # u: [batch_size, capsule_size=8, 49, 1] x capsule_units=49

    u = torch.cat(u, dim=3)
    # u: [batch_size, capsule_size=8, 49, capsule_units=49]

    u = u.view(batch_size, self.capsule_size, -1)
    # u: [batch_size, capsule_size=8, 2401=49*49]

    u = u.transpose(1, 2)
    # u: [batch_size, 2401, capsule_size=8]

    u_squashed = squash(u, dim=2)
    # u_squashed: [batch_size, 2401, capsule_size=8]

    return u_squashed


class LinearCaps(nn.Module):
  def __init__(self, routing_iters=3):
    super(LinearCaps, self).__init__()

    self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    self.routing_iters = routing_iters

    self.in_capsules = 2401
    self.in_capsule_size = 8
    self.out_capsules = 8
    self.out_capsule_size = 16

    self.W = nn.Parameter(
      torch.Tensor(
        self.in_capsules,
        self.out_capsules,
        self.out_capsule_size,
        self.in_capsule_size
      )
    )
    # W: [in_capsules, out_capsules, out_capsule_size, in_capsule_size] = [2401, 8, 16, 8]
    self.reset_parameters()

  def reset_parameters(self):
    """ Reset W.
    """
    stdv = 1. / math.sqrt(self.in_capsules)
    self.W.data.uniform_(-stdv, stdv)

  # FIXME, write in an easier way to understand, some tensors have some redundant dimensions.
  def forward(self, x):
    # x: [batch_size, in_capsules=2401, in_capsule_size=8]
    batch_size = x.size(0)

    x = torch.stack([x] * self.out_capsules, dim=2)
    # x: [batch_size, in_capsules=2401, out_capsules=8, in_capsule_size=8]

    W = torch.cat([self.W.unsqueeze(0)] * batch_size, dim=0)
    # W: [batch_size, in_capsules=2401, out_capsules=8, out_capsule_size=16, in_capsule_size=8]

    # Transform inputs by weight matrix `W`.
    u_hat = torch.matmul(W, x.unsqueeze(4)) # matrix multiplication
    # u_hat: [batch_size, in_capsules=2401, out_capsules=8, out_capsule_size=16, 1]

    u_hat_detached = u_hat.detach()
    # u_hat_detached: [batch_size, in_capsules=2401, out_capsules=8, out_capsule_size=16, 1]
    # In forward pass, `u_hat_detached` = `u_hat`, and
    # in backward, no gradient can flow from `u_hat_detached` back to `u_hat`.

    # Initialize routing logits to zero.
    b_ij = Variable(torch.zeros(self.in_capsules, self.out_capsules, 1))
    b_ij = b_ij.to(self.device)
    # b_ij: [in_capsules=2401, out_capsules=8, 1]

    # Iterative routing.
    for iteration in range(self.routing_iters):
      # Convert routing logits to softmax.
      c_ij = F.softmax(b_ij.unsqueeze(0), dim=2)
      c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)
      # c_ij: [batch_size, in_capsules=2401, out_capsules=8, 1, 1]

      if iteration == self.routing_iters - 1:
        # Apply routing `c_ij` to weighted inputs `u_hat`.
        s_j = (c_ij * u_hat).sum(dim=1, keepdim=True) # element-wise product
        # s_j: [batch_size, 1, out_capsules=8, out_capsule_size=16, 1]

        v_j = squash(s_j, dim=3)
        # v_j: [batch_size, 1, out_capsules=8, out_capsule_size=16, 1]

      else:
        # Apply routing `c_ij` to weighted inputs `u_hat`.
        s_j = (c_ij * u_hat_detached).sum(dim=1, keepdim=True) # element-wise product
        # s_j: [batch_size, 1, out_capsules=8, out_capsule_size=16, 1]

        v_j = squash(s_j, dim=3)
        # v_j: [batch_size, 1, out_capsules=8, out_capsule_size=16, 1]

        # Compute inner products of 2 16D-vectors, `u_hat` and `v_j`.
        u_vj1 = torch.matmul(u_hat_detached.transpose(3, 4), v_j).squeeze(4).mean(dim=0, keepdim=False)
        # u_vj1: [in_capsules=2401, out_capsules=8, 1]

        # Update b_ij (routing).
        b_ij = b_ij + u_vj1

    return v_j.squeeze(4).squeeze(1) # [batch_size, out_capsules=8, out_capsule_size=16]
