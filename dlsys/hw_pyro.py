# http://pyro.ai/examples/intro_part_i.html
import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist

mu = Variable(torch.zeros(1))   # mean zero
sigma = Variable(torch.ones(1)) # unit variance
x = dist.normal(mu, sigma)      # x is a sample from N(0,1)
print(x)

x = pyro.sample("my_sample", dist.normal, mu, sigma)

tnsr = torch.FloatTensor([[1,2], [3,4]])

log_p_x = dist.normal.log_pdf(x, mu, sigma)
print(log_p_x)
