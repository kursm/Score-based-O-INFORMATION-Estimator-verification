"""Fixed validation-objective plateau rule; never uses test or true information."""
import math
import torch


class PlateauMonitor:
    def __init__(self, settings, state=None):
        self.settings = settings
        self.state = state or dict(smoothed=None, best=None, stale=0, checks=0, converged=False)

    def update(self, step, loss):
        if not math.isfinite(loss):
            raise FloatingPointError("Nonfinite convergence validation objective")
        state, cfg = self.state, self.settings
        previous = state["smoothed"]
        state["smoothed"] = loss if previous is None else cfg["smoothing"]*loss+(1-cfg["smoothing"])*previous
        state["checks"] += 1
        if step < cfg["min_steps"]:
            state["best"], state["stale"] = state["smoothed"], 0
        else:
            margin = cfg["relative_tolerance"]*max(abs(state["best"] or 0.), 1e-6)
            if state["best"] is None or state["smoothed"] < state["best"]-margin:
                state["best"], state["stale"] = state["smoothed"], 0
            else:
                state["stale"] += 1
            state["converged"] = state["stale"] >= cfg["patience"]
        return state["converged"]


@torch.no_grad()
def validation_objective(model, validation, batch_size, seed, sample_count):
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    samples = validation[:sample_count]
    total = 0.
    model.eval()
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start+batch_size].to(device)
        value = model.loss(batch, generator)
        total += float(value)*len(batch)
    return total/len(samples)
