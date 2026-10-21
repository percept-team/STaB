import torch
from torch import nn


class BrownianBridge(nn.Module):
    def __init__(self, steps=8, sigma=1.0, gamma=1.0):
        super().__init__()
        if steps < 2 or sigma <= 0 or gamma <= 0:
            raise ValueError("steps >= 2, sigma > 0, gamma > 0 required")
        grid = torch.linspace(0, 1, steps + 1).pow(gamma)
        weights = torch.empty(steps)
        weights[0] = 0.5
        weights[1:] = (grid[2:] - grid[1:-1]) / (2 * sigma**2 * grid[1:-1] * grid[2:])
        self.register_buffer("grid", grid)
        self.register_buffer("probabilities", weights / weights.sum())
        self.sigma = sigma

    def corrupt(self, target, endpoint, index, generator=None):
        tau = self.grid[index + 1].to(target)
        noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
        return (1 - tau) * target + tau * endpoint + self.sigma * (tau * (1 - tau)).sqrt() * noise

    def sample_index(self, generator=None):
        return int(torch.multinomial(self.probabilities, 1, generator=generator).item())

    def reverse_parameters(self, state, prediction, index):
        if index < 1 or index >= len(self.grid) - 1:
            raise ValueError("reverse transition index must be in [1, K-1]")
        tau, previous = self.grid[index + 1], self.grid[index]
        ratio = (previous / tau).to(state)
        mean = (1 - ratio) * prediction + ratio * state
        variance = self.sigma**2 * previous * (tau - previous) / tau
        return mean, variance.to(state)

    @torch.no_grad()
    def sample(self, endpoint, predict, generator=None):
        state = endpoint.clone()
        for index in range(len(self.grid) - 2, -1, -1):
            prediction = predict(state, self.grid[index + 1].to(state))
            if index == 0:
                return prediction
            mean, variance = self.reverse_parameters(state, prediction, index)
            noise = torch.randn(state.shape, device=state.device, dtype=state.dtype, generator=generator)
            state = mean + variance.sqrt() * noise
