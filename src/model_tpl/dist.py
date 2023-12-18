import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Function

eps=1e-8


class STHeaviside(Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.zeros(x.size()).type_as(x)
        y[x >= 0] = 1
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class NormalDistUtil(object):
    @staticmethod
    def log_density(X, MU, LOGVAR):
        """ compute log pdf of normal distribution

        Args:
            X (_type_): sample point
            MU (_type_): mu of normal dist
            LOGVAR (_type_): logvar of normal dist
        """
        norm = - 0.5 * (math.log(2 * math.pi) + LOGVAR)
        log_density = norm - 0.5 * ((X - MU).pow(2) * torch.exp(-LOGVAR))
        return log_density

    @staticmethod
    def kld(MU:float, LOGVAR:float, mu_move):
        """compute KL divergence between X and Normal Dist whose (mu, var) equals to (mu_move, 1)

        Args:
            MU (float): _description_
            VAR (float): _description_
            mu_move (_type_): _description_
        """

        return 0.5 * (LOGVAR.exp() - LOGVAR + MU.pow(2) - 2 * mu_move * MU + mu_move ** 2 - 1)

    @staticmethod
    def sample(mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        return mu + std * eps


class BernoulliUtil(nn.Module):
    """Samples from a Bernoulli distribution where the probability is given
    by the sigmoid of the given parameter.
    """

    def __init__(self, p=0.5, stgradient=False):
        super().__init__()
        p = torch.Tensor([p])
        self.p = torch.log(p / (1 - p) + eps)
        self.stgradient = stgradient

    def _check_inputs(self, size, ps):
        if size is None and ps is None:
            raise ValueError(
                'Either one of size or params should be provided.')
        elif size is not None and ps is not None:
            if ps.ndimension() > len(size):
                return ps.squeeze(-1).expand(size)
            else:
                return ps.expand(size)
        elif size is not None:
            return self.p.expand(size)
        elif ps is not None:
            return ps
        else:
            raise ValueError(
                'Given invalid inputs: size={}, ps={})'.format(size, ps))

    def _sample_logistic(self, size):
        u = torch.rand(size)
        l = torch.log(u + eps) - torch.log(1 - u + eps)
        return l

    def default_sample(self, size=None, params=None):
        presigm_ps = self._check_inputs(size, params)
        logp = F.logsigmoid(presigm_ps)
        logq = F.logsigmoid(-presigm_ps)
        l = self._sample_logistic(logp.size()).type_as(presigm_ps)
        z = logp - logq + l
        b = STHeaviside.apply(z)
        return b if self.stgradient else b.detach()

    def sample(self, size=None, params=None,type_='gumbel_softmax', **kwargs):
        if type_ == 'default':
            return self.default_sample(size, params)
        elif type_ == 'gumbel_softmax':
            tau = kwargs.get('tau', 1.0)
            hard = kwargs.get('hard', True)
            ext_params = torch.log(torch.stack([1 - params, params],dim=2) + eps)
            return F.gumbel_softmax(logits=ext_params, tau=tau, hard=hard)[:,:,-1]
        else:
            raise ValueError(f"Unknown Type of sample: {type_}")

    def log_density(self, sample, params=None, is_check=True):
        if is_check:
            presigm_ps = self._check_inputs(sample.size(), params).type_as(sample)
        else:
            presigm_ps = params
        p = (torch.sigmoid(presigm_ps) + eps) * (1 - 2 * eps)
        logp = sample * torch.log(p + eps) + (1 - sample) * torch.log(1 - p + eps)
        return logp

    def get_params(self):
        return self.p

    @property
    def nparams(self):
        return 1

    @property
    def ndim(self):
        return 1

    @property
    def is_reparameterizable(self):
        return self.stgradient

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' ({:.3f})'.format(
            torch.sigmoid(self.p.data)[0])
        return tmpstr
