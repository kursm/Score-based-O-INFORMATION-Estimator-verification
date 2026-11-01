"""Adapters see only X and standard method inputs; U/V is used only in diagnostics."""
from copy import deepcopy
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from ._upstream_mlp import UnetMLP_simple
from .sde import VPSDE, training_masks


def randn_like(x, generator):
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)


class SOI(nn.Module):
    def __init__(self, n, block_dim, params):
        super().__init__()
        self.n, self.block_dim, self.params_config = n, block_dim, params
        dimension = n*block_dim
        width = params["width"] or (128 if dimension <= 50 else 192 if dimension <= 100 else 256)
        time_dim = params["time_dim"] or width
        self.score = UnetMLP_simple(dimension, init_dim=width, time_dim=time_dim,
                                   nb_var=n, dim_mults=params["dim_mults"],
                                   resnet_block_groups=params["groups"])
        self.ema = deepcopy(self.score).requires_grad_(False) if params["use_ema"] else None
        self.sde = VPSDE(**params["sde"])
        self.register_buffer("masks", training_masks(n, params["weight_s_functions"]))
        self.architecture = dict(width=width, time_dim=time_dim, dim_mults=params["dim_mults"],
                                 registered_parameters=sum(p.numel() for p in self.score.parameters()),
                                 note="Official constructor dim_mults=[]; mid_block1 is registered but unused.")

    def predict(self, x, t, mask, inference=False):
        net = self.ema if inference and self.ema is not None else self.score
        mask = mask.expand(x.shape[0], -1)
        tau = t*mask.clamp(0, 1)+(mask < 0).to(t.dtype)
        return net(x, t=tau)

    def loss_fixed(self, x, t, noise, mask, marginal_noise):
        flat = x.flatten(1)
        expanded = mask.repeat_interleave(self.block_dim, dim=1)
        active, missing = expanded.clamp(0, 1), (expanded < 0).float()
        noised = self.sde.perturb(flat, t, noise)
        inp = (active*noised+(1-active)*flat)*(1-missing)+missing*marginal_noise
        prediction = self.predict(inp, t, mask)
        weight = 2-active.mean(dim=1, keepdim=True)
        return (weight*(prediction-noise).square()*active).sum(1).mean()

    def loss(self, x, generator):
        t = self.sde.sample_time(len(x), x.device, generator)
        indices = torch.randint(len(self.masks), (len(x),), device=x.device, generator=generator)
        flat = x.flatten(1)
        return self.loss_fixed(x, t, randn_like(flat, generator), self.masks[indices],
                               randn_like(flat, generator))

    @torch.no_grad()
    def update_ema(self):
        if self.ema is not None:
            # The original EMA traverses state_dict, preserving repeated aliases
            # (proj is also final_lin.2). Its output head is therefore updated twice.
            decay = self.params_config["ema_decay"]
            for e, p in zip(self.ema.state_dict().values(), self.score.state_dict().values()):
                e.copy_(decay*e+(1-decay)*p)

    def trainable_parameters(self):
        return self.score.parameters()

    @torch.no_grad()
    def evaluate(self, data, eval_batch_size, generator, u_dim=None, mc_iter=None):
        self.eval()
        device = next(self.parameters()).device
        count = len(data)
        iterations = mc_iter or self.params_config["mc_iter"]
        tc_coord = torch.zeros((self.n, self.block_dim), dtype=torch.float64, device=device)
        dtc_coord = torch.zeros_like(tc_coord)
        for _ in range(iterations):
            for start in range(0, count, eval_batch_size):
                x = data[start:start+eval_batch_size].to(device)
                flat = x.flatten(1)
                t = self.sde.sample_time(len(x), device, generator)
                xt = self.sde.perturb(flat, t, randn_like(flat, generator))
                joint = self.predict(xt, t, torch.ones((1, self.n), device=device), True)
                marginal, conditional = [], []
                for i in range(self.n):
                    lo, hi = i*self.block_dim, (i+1)*self.block_dim
                    mask = torch.zeros((1, self.n), device=device)
                    mask[:, i] = 1
                    xm = randn_like(flat, generator)
                    xm[:, lo:hi] = xt[:, lo:hi]
                    xc = flat.clone()
                    xc[:, lo:hi] = xt[:, lo:hi]
                    marginal.append(self.predict(xm, t, 2*mask-1, True)[:, lo:hi])
                    conditional.append(self.predict(xc, t, mask, True)[:, lo:hi])
                weight = self.sde.integral_weight(t)
                tc_coord += (weight*(joint.double()-torch.cat(marginal, 1).double()).square()
                             ).sum(0).reshape(self.n, self.block_dim)
                dtc_coord += (weight*(joint.double()-torch.cat(conditional, 1).double()).square()
                              ).sum(0).reshape(self.n, self.block_dim)
        tc_coord /= count*iterations
        dtc_coord /= count*iterations
        tc, dtc = tc_coord.sum().item(), dtc_coord.sum().item()
        out = dict(tc_hat=tc, dtc_hat=dtc, omega_hat=tc-dtc,
                   evaluation_mode="vp_"+self.sde.inference_weighting,
                   tc_coordinates=tc_coord.cpu().tolist(),
                   dtc_coordinates=dtc_coord.cpu().tolist())
        if u_dim is not None:
            out.update(tc_u=tc_coord[:, :u_dim].sum().item(), tc_v=tc_coord[:, u_dim:].sum().item(),
                       dtc_u=dtc_coord[:, :u_dim].sum().item(), dtc_v=dtc_coord[:, u_dim:].sum().item())
        return out

    @torch.no_grad()
    def regression_diagnostics(self, data, batch_size, generator, u_dim):
        """Unweighted per-coordinate noise MSE, separately by regime and t bin."""
        self.eval()
        device = next(self.parameters()).device
        result = []
        for regime in ["joint", "marginal", "conditional"]:
            for left, right in zip([0., .001, .01, .1, .5], [.001, .01, .1, .5, 1.]):
                sums = {"u": 0., "v": 0.}
                counts = {"u": 0, "v": 0}
                for start in range(0, len(data), batch_size):
                    x = data[start:start+batch_size].to(device)
                    flat = x.flatten(1)
                    t = left+(right-left)*torch.rand((len(x), 1), device=device, generator=generator)
                    noise = randn_like(flat, generator)
                    xt = self.sde.perturb(flat, t, noise)
                    targets = range(self.n) if regime != "joint" else [None]
                    for i in targets:
                        mask = torch.ones((1, self.n), device=device)
                        inp = xt
                        if i is not None:
                            mask.fill_(-1 if regime == "marginal" else 0)
                            mask[:, i] = 1
                            inp = randn_like(flat, generator) if regime == "marginal" else flat.clone()
                            inp[:, i*self.block_dim:(i+1)*self.block_dim] = xt[:, i*self.block_dim:(i+1)*self.block_dim]
                        error = (self.predict(inp, t, mask, True)-noise).double().square().reshape(
                            len(x), self.n, self.block_dim)
                        if i is not None:
                            error = error[:, i:i+1]
                        for name, selection in [("u", error[:, :, :u_dim]),
                                                ("v", error[:, :, u_dim:])]:
                            sums[name] += selection.sum().item()
                            counts[name] += selection.numel()
                result.append(dict(regime=regime, t_left=left, t_right=right,
                                   **{name+"_mse": sums[name]/counts[name] if counts[name] else None
                                      for name in sums}))
        return result


class Critic(nn.Module):
    def __init__(self, x_dim, y_dim, width, kind, hidden_layers=None):
        super().__init__()
        self.kind = kind
        depth = hidden_layers if hidden_layers is not None else (1 if kind == "mine" else 2)
        layers = []
        dim = x_dim+y_dim
        for _ in range(depth):
            layers += [nn.Linear(dim, width), nn.ReLU()]
            dim = width
        layers += [nn.Linear(dim, 1)]
        self.net = nn.Sequential(*layers)

    def joint(self, x, y):
        return self.net(torch.cat((x, y), -1)).flatten()

    def pair_logsumexp(self, x, y, chunk_size, recompute=True):
        """All ordered pairs, including diagonal. Max chunk_size pair activations in memory."""
        nx, ny = len(x), len(y)
        chunk_size = chunk_size or nx*ny
        total = None
        for start in range(0, nx*ny, chunk_size):
            ids = torch.arange(start, min(start+chunk_size, nx*ny), device=x.device)
            a, b = x[ids % nx], y[ids // nx]
            def reduce_chunk(a, b):
                return torch.logsumexp(self.joint(a, b).double()-1., dim=0)
            if recompute and torch.is_grad_enabled() and nx*ny > chunk_size:
                part = checkpoint(reduce_chunk, a, b, use_reentrant=False)
            else:
                part = reduce_chunk(a, b)
            total = part if total is None else torch.logaddexp(total, part)
        return total

    def statistics(self, x, y, generator=None, pair_chunk=4096, negative_sampling="with_replacement"):
        positive = self.joint(x, y).double().mean()
        if self.kind == "mine":
            if negative_sampling == "permutation":
                ids = torch.randperm(len(y), device=y.device, generator=generator)
            else:
                ids = torch.randint(len(y), (len(y),), device=y.device, generator=generator)
            negative_logmean = torch.logsumexp(self.joint(x, y[ids]).double(), 0)-math.log(len(y))
        else:
            negative_logmean = self.pair_logsumexp(x, y, pair_chunk)-math.log(len(x)*len(y))
        return positive, negative_logmean

    def bound(self, x, y, **kwargs):
        p, q = self.statistics(x, y, **kwargs)
        return p-q if self.kind == "mine" else p-q.exp()


class Line(nn.Module):
    def __init__(self, n, block_dim, params, kind):
        super().__init__()
        self.n, self.block_dim, self.params_config, self.kind = n, block_dim, params, kind
        hidden = params["width"] or (8 if block_dim == 1 else 16 if block_dim <= 10 else 20)
        cap = params["width_cap"] or (32 if kind == "mine" else 256)
        # Preserve original initialization order: S critics, then TC critics.
        self.pairs = [(tuple(j for j in range(n) if j != i), (i,), "s_"+str(i)) for i in range(n)]
        self.pairs += [(tuple(range(i+1)), (i+1,), "tc_"+str(i)) for i in range(n-1)]
        self.critics = nn.ModuleList([Critic(len(a)*block_dim, len(b)*block_dim,
            min(hidden*(n-1 if name.startswith("s_") else len(a)), cap), kind,
            params["hidden_layers"]) for a, b, name in self.pairs])
        self.architecture = dict(base_width=hidden, cap=cap,
            hidden_layers=params["hidden_layers"] or (1 if kind == "mine" else 2),
            critics=len(self.critics), shared_weights=False, duplicate_mi_terms="independent critics")

    def split(self, x, pair):
        a, b, _ = pair
        return x[:, a, :].flatten(1), x[:, b, :].flatten(1)

    def loss(self, x, generator):
        args = dict(generator=generator, pair_chunk=self.params_config["pair_chunk"],
                    negative_sampling=self.params_config["negative_sampling"])
        # Upstream draws training negatives for TC critics first, then S critics.
        order = list(range(self.n, len(self.critics)))+list(range(self.n))
        bounds = [self.critics[i].bound(*self.split(x, self.pairs[i]), **args) for i in order]
        return -torch.stack(bounds).mean()

    def trainable_parameters(self):
        return self.parameters()

    def update_ema(self):
        pass

    @torch.no_grad()
    def evaluate(self, data, eval_batch_size, generator, **kwargs):
        self.eval()
        device = next(self.parameters()).device
        terms = {}
        for critic, pair in zip(self.critics, self.pairs):
            stats = []
            for start in range(0, len(data), eval_batch_size):
                x = data[start:start+eval_batch_size].to(device)
                p, logq = critic.statistics(*self.split(x, pair), generator=generator,
                    pair_chunk=self.params_config["pair_chunk"],
                    negative_sampling=self.params_config["negative_sampling"])
                stats.append((len(x), p, logq))
            total = sum(s[0] for s in stats)
            if self.params_config["eval_aggregation"] == "paper_batch_mean":
                bound = torch.stack([p-q if self.kind == "mine" else p-q.exp()
                                     for _, p, q in stats]).mean()
            else:
                positive = sum(n*p for n, p, _ in stats)/total
                logq = torch.logsumexp(torch.stack(
                    [q+math.log(n/total) for n, _, q in stats]), 0)
                bound = positive-logq if self.kind == "mine" else positive-logq.exp()
            terms[pair[2]] = bound.item()
        tc = sum(v for k, v in terms.items() if k.startswith("tc_"))
        s = sum(v for k, v in terms.items() if k.startswith("s_"))
        return dict(tc_hat=tc, dtc_hat=s-tc, omega_hat=2*tc-s, mi_terms=terms,
                    evaluation_mode=self.params_config["eval_aggregation"],
                    product_scheme="within fixed evaluation batches")


def make_estimator(method, n, block_dim, params):
    return SOI(n, block_dim, params) if method == "soi" else Line(
        n, block_dim, params, "mine" if method == "line_mine" else "nwj")
