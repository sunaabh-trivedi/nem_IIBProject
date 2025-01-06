
import torch
import numpy as np
from functools import partial
import copy
from torch.distributions import Normal, Gumbel


class LangevinDynamics(object):

    def __init__(self,
                 x: torch.Tensor,
                 energy_func: callable,
                 step_size: float,
                 mh: bool = True,
                 device: str = 'cpu',
                 point_estimator: bool = False):
        super(LangevinDynamics, self).__init__()

        self.x = x
        self.step_size = step_size
        self.energy_func = energy_func
        self.mh= mh
        self.device = device
        self.point_estimator = point_estimator

        if self.mh:
            x_c = self.x.detach()
            x_c.requires_grad = True
            f_xc = self.energy_func(x_c)
            grad_xc = torch.autograd.grad(f_xc.sum(), x_c,create_graph=False)[0]

            self.f_x = f_xc.detach()
            self.grad_x = grad_xc.detach()

    def sample(self) -> tuple:
        if self.point_estimator == True:
            x_c = self.x.detach()
            x_c.requires_grad = True
            f_xc = self.energy_func(x_c)
            grad_xc = torch.autograd.grad(f_xc.sum(), x_c,create_graph=False)[0]
            x_p = x_c - self.step_size * grad_xc 
            self.x = x_p.detach()
            return copy.deepcopy(x_p.detach()), None

        if self.mh == False:
            x_c = self.x.detach()
            x_c.requires_grad = True
            f_xc = self.energy_func(x_c)
            grad_xc = torch.autograd.grad(f_xc.sum(), x_c,create_graph=False)[0]

            x_p = x_c - self.step_size * grad_xc + torch.sqrt(torch.tensor(2.0*self.step_size, device=self.device)) * torch.randn_like(x_c, device=self.device)
            
            self.x = x_p.detach()
            return copy.deepcopy(x_p.detach()), f_xc.detach()
        
        else:
            x_c = self.x.detach()
            f_xc = self.f_x.detach()
            grad_xc = self.grad_x.detach()
            
            x_p = x_c - self.step_size * grad_xc + torch.sqrt(torch.tensor(2.0*self.step_size, device=self.device)) * torch.randn_like(self.x, device=self.device)
            x_p = x_p.detach()
            x_p.requires_grad = True
            f_xp = self.energy_func(x_p)
            grad_xp = torch.autograd.grad(f_xp.sum(), x_p,create_graph=False)[0]
            log_joint_prob_2 = -f_xc-torch.norm(x_p-x_c+self.step_size * grad_xc, dim=-1)**2/(4*self.step_size)
            log_joint_prob_1 = -f_xp-torch.norm(x_c-x_p+self.step_size * grad_xp, dim=-1)**2/(4*self.step_size)

            log_accept_rate = log_joint_prob_1 - log_joint_prob_2
            is_accept = torch.rand_like(log_accept_rate).log() <= log_accept_rate
            is_accept = is_accept.unsqueeze(-1)

            self.x = torch.where(is_accept, x_p.detach(), self.x)
            self.f_x = torch.where(is_accept.squeeze(-1), f_xp.detach(), self.f_x)
            self.grad_x = torch.where(is_accept, grad_xp.detach(), self.grad_x)  

            acc_rate = torch.minimum(torch.ones_like(log_accept_rate), log_accept_rate.exp()).mean()
            
            return copy.deepcopy(self.x.detach()), acc_rate.item()
        



def target_density_and_grad_fn_full(x, inv_temperature, target_log_prob_fn):
    x = x.clone().detach().requires_grad_(True)
    log_prob = target_log_prob_fn(x) * inv_temperature
    log_prob_sum = log_prob.sum()
    log_prob_sum.backward()
    grad = x.grad.clone().detach()
    return log_prob.detach(), grad


class HamiltonianMonteCarlo(object):

    def __init__(self,
                 x,
                 energy_func: callable,
                 step_size: float,
                 num_leapfrog_steps_per_hmc_step: int,
                 inv_temperature: float = 1.0,
                 device: str = 'cpu'):
        super(HamiltonianMonteCarlo, self).__init__()

        self.x = x
        self.step_size = step_size
        self.target_density_and_grad_fn = partial(target_density_and_grad_fn_full, target_log_prob_fn=lambda x: -energy_func(x))
        self.device = device
        self.inv_temperature = inv_temperature
        self.num_leapfrog_steps_per_hmc_step = num_leapfrog_steps_per_hmc_step

        self.current_log_prob, self.current_grad = self.target_density_and_grad_fn(x, self.inv_temperature)

    def leapfrog_integration(self, p):
        """
        Leapfrog integration for simulating Hamiltonian dynamics.
        """
        x = self.x.detach().clone()
        p = p.detach().clone()

        # Half step for momentum
        p += 0.5 * self.step_size * self.current_grad

        # Full steps for position
        for _ in range(self.num_leapfrog_steps_per_hmc_step - 1):
            x += self.step_size * p
            _, grad = self.target_density_and_grad_fn(x, self.inv_temperature)
            p += self.step_size * grad  # this combines two half steps for momentum

        # Final update of position and half step for momentum
        x += self.step_size * p
        new_log_prob, new_grad = self.target_density_and_grad_fn(x, self.inv_temperature)
        p += 0.5 * self.step_size * new_grad

        return x, p, new_log_prob, new_grad


    def sample(self):
        """
        Hamiltonian Monte Carlo step.
        """

        # Sample a new momentum
        p = torch.randn_like(self.x, device=self.device)

        # Simulate Hamiltonian dynamics
        new_x, new_p, new_log_prob, new_grad = self.leapfrog_integration(p)

        # Hamiltonian (log probability + kinetic energy)
        current_hamiltonian = self.current_log_prob - 0.5 * p.pow(2).sum(-1)
        new_hamiltonian = new_log_prob - 0.5 * new_p.pow(2).sum(-1)
        
        log_accept_rate = -current_hamiltonian + new_hamiltonian
        is_accept = torch.rand_like(log_accept_rate, device=self.device).log() < log_accept_rate
        is_accept = is_accept.unsqueeze(-1)

        self.x = torch.where(is_accept, new_x.detach(), self.x)
        self.current_grad = torch.where(is_accept, new_grad.detach(), self.current_grad)
        self.current_log_prob = torch.where(is_accept.squeeze(-1), new_log_prob.detach(), self.current_log_prob)

        acc_rate = torch.minimum(torch.ones_like(log_accept_rate), log_accept_rate.exp()).mean()
        
        return copy.deepcopy(self.x.detach()), acc_rate.item()

def AIS(ais_step: int, 
        smc_gap: int,
        hmc_step: int, 
        hmc_step_size: float, 
        x_importance_sample: any, 
        proposal_log_p: callable, 
        target_log_p: callable,
        verbose: bool,
        device: str,
        LG: bool = False, # if LG, then run LG as the kernel; otherwise, run HMC as the kernel.
        LG_step: int = 1,
        ):
    final_target_log_p = target_log_p
    initial_proposal_log_p = proposal_log_p

    is_target = lambda x: 1 / ais_step * final_target_log_p(x) + (1 - 1 / ais_step) * initial_proposal_log_p(x)      
    is_weights = is_target(x_importance_sample) - initial_proposal_log_p(x_importance_sample)
    acc = []
    for step in range(ais_step-1):

        if (step + 1) % smc_gap == 0:
        

            # perform SMC by batched gumbel max trick
            is_weight_old = is_weights.clone()
            is_weights = is_weights[None, :, :]
            # print(is_weights.shape)
            is_weights = is_weights.repeat([is_weights.shape[1], 1, 1]) # ss, sample_size, bzs
            gumbeled_density_ratio = is_weights + Gumbel(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)).sample(is_weights.shape)
            idx = gumbeled_density_ratio.argmax(0).T # bzs, sample_size
            # print(idx.shape)
            x_importance_sample = x_importance_sample.permute(1, 0, 2) # ss, bsz, dim -> bsz, ss, dim
            # Add an extra dimension to `indices` to match the shape for gathering
            expanded_indices = idx.unsqueeze(-1).expand(-1, -1, x_importance_sample.shape[-1])  # Shape: (N, D, v)
            x_importance_sample = torch.gather(x_importance_sample, 1, expanded_indices)
            x_importance_sample = x_importance_sample.permute(1, 0, 2) # ss, bsz, dim <- bsz, ss, dim
            is_weights = 0 * is_weight_old
        


        s = (step+1) / ais_step
        s_next = (step+2) / ais_step

        # use HMC for pi \propto proposal^(1-s) target^s
        target = lambda x: s * final_target_log_p(x) + (1-s) * initial_proposal_log_p(x) # HMC target
        if not LG:
            hmc = HamiltonianMonteCarlo(x_importance_sample.clone(), energy_func=lambda x: -target(x), step_size=hmc_step_size, num_leapfrog_steps_per_hmc_step=hmc_step, device=device) 
            x_importance_sample, rate = hmc.sample()
            x_importance_sample = x_importance_sample.detach()
            if verbose:
                print('AIS step %.2f'%s, 'HMC Acc rate %.3f'%rate, 'IS sample quantile:', x_importance_sample.min().item(), x_importance_sample.quantile(0.25).item(),  x_importance_sample.quantile(0.75).item(), x_importance_sample.max().item())
        else:
            lg = LangevinDynamics(x_importance_sample.clone(),
                                  energy_func=lambda x: -target(x),
                                  step_size=hmc_step_size,
                                  mh=True,
                                  device=device)
            for _ in range(LG_step):
                x_importance_sample, rate = lg.sample()
                x_importance_sample = x_importance_sample.detach()
            if verbose:
                print('AIS step %.2f'%s, 'LG Acc rate %.3f'%rate, 'IS sample quantile:', x_importance_sample.min().item(), x_importance_sample.quantile(0.25).item(),  x_importance_sample.quantile(0.75).item(), x_importance_sample.max().item())
        acc.append(rate)

            
        # calculate the IS weight
        is_target = lambda x: s_next * final_target_log_p(x) + (1-s_next) * initial_proposal_log_p(x)
        is_weights += (is_target(x_importance_sample) - target(x_importance_sample))

        
    return is_weights, x_importance_sample, acc