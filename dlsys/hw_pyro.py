# http://pyro.ai/examples/intro_part_i.html
import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist

mu = Variable(torch.zeros(1))   # mean zero
sigma = Variable(torch.ones(1)) # unit variance
x = dist.normal(mu, sigma)      # x is a sample from N(0,1)

# Pyro’s backend uses these names to uniquely identify sample statements and
# change their behavior at runtime depending on how the enclosing stochastic
# function is being used
x = pyro.sample("testing_sample", dist.normal, mu, sigma)

tnsr = torch.FloatTensor([[1,2], [3,4]])
log_p_x = dist.normal.log_pdf(x, mu, sigma)

def weather():
    """
    The function specifies a joint probability distribution over
    two named random variables: cloudy and temp. As such, it defines
    a probabilistic model that we can reason about using
    the techniques of probability theory.
    """
    cloudy = pyro.sample('cloudy', dist.bernoulli,
                         Variable(torch.Tensor([0.3])))
    cloudy = 'cloudy' if cloudy.data[0] == 1.0 else 'sunny'
    mean_temp = {'cloudy': [55.0], 'sunny': [75.0]}[cloudy]
    sigma_temp = {'cloudy': [10.0], 'sunny': [15.0]}[cloudy]
    temp = pyro.sample('temp', dist.normal,
                       Variable(torch.Tensor(mean_temp)),
                       Variable(torch.Tensor(sigma_temp)))
    return cloudy, temp.data[0]

list(map(lambda _: weather(), range(7)))
