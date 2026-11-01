"""VP-SDE and the official truncated importance distribution."""
import math
import torch
import torch.nn.functional as F


class VPSDE:
    def __init__(self, beta_min=.1, beta_max=20., t_epsilon=.001,
                 importance_sampling=True, inference_weighting="upstream_constant", uniform_epsilon=1e-5):
        if not (0 < beta_min <= beta_max and 0 < t_epsilon < 1):
            raise ValueError("Invalid VP-SDE parameters")
        self.beta_min, self.beta_max = beta_min, beta_max
        if not 0 < uniform_epsilon < 1:
            raise ValueError("Invalid uniform time epsilon")
        self.uniform_epsilon = uniform_epsilon
        self.t_epsilon, self.importance_sampling = t_epsilon, importance_sampling
        self.inference_weighting = inference_weighting
        e = t_epsilon
        self.r_eps = (beta_min+(beta_max-beta_min)*e)/(-math.expm1(-self.integral(e)))
        self.ant_eps = self.antiderivative(e)
        self.normalizer = self.r_eps*e + self.antiderivative(1.)-self.ant_eps

    def integral(self, t):
        return self.beta_min*t + .5*(self.beta_max-self.beta_min)*t*t

    def antiderivative(self, t):
        b = self.integral(t)
        return math.log(-math.expm1(-b))+b

    def beta(self, t):
        return self.beta_min+(self.beta_max-self.beta_min)*t

    def mean_std(self, t):
        b = self.integral(t)
        return torch.exp(-.5*b), torch.sqrt(-torch.expm1(-b))

    def inv_cdf(self, u):
        # Use float64 for inverse CDF, rationalized quadratic root.
        u = u.double()
        value = self.normalizer*u + self.ant_eps-self.r_eps*self.t_epsilon
        integral_beta = F.softplus(value)
        delta = self.beta_max-self.beta_min
        upper = 2*integral_beta/(torch.sqrt(
            self.beta_min**2 + 2*delta*integral_beta)+self.beta_min)
        return torch.where(u <= self.t_epsilon*self.r_eps/self.normalizer,
                           u*self.normalizer/self.r_eps, upper)

    def sample_time(self, count, device, generator):
        u = torch.rand((count, 1), device=device, generator=generator, dtype=torch.float64)
        u = u.clamp_min(torch.finfo(torch.float64).eps)
        if self.importance_sampling:
            return self.inv_cdf(u).float()
        return (self.uniform_epsilon+(1-self.uniform_epsilon)*u).float()

    def perturb(self, x, t, noise):
        mean, std = self.mean_std(t)
        return mean*x+std*noise

    def integral_weight(self, t):
        if self.importance_sampling:
            if self.inference_weighting == "upstream_constant":
                return torch.full_like(t, .5*self.normalizer, dtype=torch.float64)
            # Optional unbiased correction for the floored importance pdf.
            _, std = self.mean_std(t.double())
            r = self.beta(t.double())/std.square()
            return .5*self.normalizer*torch.where(t < self.t_epsilon,
                                                 r/self.r_eps, torch.ones_like(r))
        _, std = self.mean_std(t.double())
        return .5*(1-self.uniform_epsilon)*self.beta(t.double())/std.square()


def training_masks(n, weighted=True):
    masks = [([1]*n, n)]
    for i in range(n):
        marginal = [-1]*n
        conditional = [0]*n
        marginal[i] = conditional[i] = 1
        masks += [(marginal, 1), (conditional, 1+(n-1)//2)]
    order = {0: 0, 1: 1, -1: 2}
    masks.sort(key=lambda item: tuple(order[x] for x in item[0]))
    return torch.tensor([m for m, weight in masks for _ in range(weight if weighted else 1)],
                        dtype=torch.float32)
