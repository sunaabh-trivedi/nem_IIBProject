from contextlib import contextmanager

import numpy as np
import torch

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.models.components.sdes import VEReverseSDE
from dem.utils.data_utils import remove_mean


@contextmanager
def conditional_no_grad(condition):
    if condition:
        with torch.no_grad():
            yield
    else:
        yield


def grad_E(x, energy_function):
    with torch.enable_grad():
        x = x.requires_grad_()
        return torch.autograd.grad(torch.sum(energy_function(x)), x)[0].detach()


def negative_time_descent(x, energy_function, num_steps, dt=1e-4):
    samples = []
    for _ in range(num_steps):
        drift = grad_E(x, energy_function)
        x = x + drift * dt

        if energy_function.is_molecule:
            x = remove_mean(x, energy_function.n_particles, energy_function.n_spatial_dim)

        samples.append(x)
    return torch.stack(samples)


def euler_maruyama_step(
    sde: VEReverseSDE, t: torch.Tensor, x: torch.Tensor, dt: float, var_preserve, diffusion_scale=1.0
):
    # Calculate drift and diffusion terms
    drift = sde.f(t, x) * dt
    

    # Update the state
    if var_preserve:
        sigma_max = torch.full_like(t, sde.noise_schedule.sigma_max)
        lambda_k = 1 - torch.sqrt(1 - sde.noise_schedule.a(t, dt))
        x_next = torch.sqrt(1 - sde.noise_schedule.a(t, dt)) * x + \
            2 * sigma_max **2 * lambda_k * drift/dt + sigma_max * torch.sqrt(sde.noise_schedule.a(t, dt)) * torch.randn_like(x)
        drift = 2 * sigma_max **2 * lambda_k **2 / sde.noise_schedule.a(t, dt) * torch.norm(drift / dt, dim=-1)
    else:
        diffusion = diffusion_scale * sde.g(t, x) * np.sqrt(dt) * torch.randn_like(x)
        # diffusion = diffusion_scale * 1/(torch.sqrt(1-t))*np.sqrt(dt)*torch.randn_like(x)
        x_next = x + drift + diffusion
    return x_next, drift


def integrate_pfode(
    sde: VEReverseSDE,
    x0: torch.Tensor,
    num_integration_steps: int,
    reverse_time: bool = True,
):
    start_time = 1.0 if reverse_time else 0.0
    end_time = 1.0 - start_time

    times = torch.linspace(start_time, end_time, num_integration_steps + 1, device=x0.device)[:-1]

    x = x0
    samples = []
    with torch.no_grad():
        for t in times:
            x, f = euler_maruyama_step(sde, t, x, 1 / num_integration_steps)
            samples.append(x)

    return torch.stack(samples)


def integrate_sde(
    sde: VEReverseSDE,
    x0: torch.Tensor,
    num_integration_steps: int,
    energy_function: BaseEnergyFunction,
    reverse_time: bool = True,
    diffusion_scale=1.0,
    no_grad=True,
    time_range=0.999,
    negative_time=False,
    num_negative_time_steps=100,
    var_preserve=False,
    metroplolis_hasting=False
):
    start_time = time_range if reverse_time else 0.0
    end_time = time_range - start_time

    times = torch.linspace(start_time, end_time, num_integration_steps + 1, device=x0.device)[:-1]

    x = x0
    samples = []
    
    if var_preserve:#r for DDS sampler
        r_k = torch.zeros([x0.shape[0]]).to(x0.device)
    with conditional_no_grad(no_grad):
        for t in times:
            if not metroplolis_hasting:
                x, f = euler_maruyama_step(
                    sde, t, x, time_range / num_integration_steps, var_preserve,
                    diffusion_scale
                )
                if var_preserve:
                    r_k += f
            else:
                x = sde.mh_sample(t, x, time_range / num_integration_steps, diffusion_scale)
            if energy_function.is_molecule:
                x = remove_mean(x, energy_function.n_particles, energy_function.n_spatial_dim)
            if energy_function._can_normalize:
                clip_min = energy_function.normalization_min
                clip_max = energy_function.normalization_max
                x = torch.clamp(x, -1.0, 1.0)
            else:
                test_set = energy_function._val_set
                clip_range = [torch.min(test_set), torch.max(test_set)]
                x = torch.clamp(x, clip_range[0], clip_range[1])
            samples.append(x)

    assert not torch.isnan(x0).any()
    samples = torch.stack(samples)
    # Screen energies for each batch item
    '''
    for i in range(samples.shape[1]):
        if torch.isnan(samples[-1, i]).any():
            roll_back = False
            for j in range(1, samples.shape[0]):
                if not torch.isnan(samples[samples.shape[1] - j, i]).any():
                    samples[-1, i] = samples[samples.shape[1] - j, i]
                    roll_back = True
                    break
            if not roll_back:
                samples[-1, i] = x0[i]
    assert not torch.isnan(samples[-1]).any()
    '''
    if negative_time:
        print("doing negative time descent...")
        samples_langevin = negative_time_descent(
            x, energy_function, num_steps=num_negative_time_steps
        )
        samples = torch.concatenate((samples, samples_langevin), axis=0)
    if not var_preserve:
        return samples
    else:
        return samples, r_k
