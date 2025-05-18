import torch
import numpy as np
from scipy.stats import entropy

class AnnealingSchedule:
    """
    Log-Normal Annealing Schedule for sampling times in training loop
    ln(t_lognorm) ~ N(mu,sigma^2)
    t_uniform ~ U(0,1)

    Anneals towards uniform distribution:
    t = (1-alpha^beta)*t_lognorm + alpha^beta*t_uniform

    sigma_t = sigma(t)
    x_t = x_0 + sigma_t*N(0, 1)
    """

    def __init__(self, beta: float, n: int) -> None:
        self.log_mean_start = -5.0
        self.log_std_start = 2

        self.num_epochs_to_uniform = n

        self.beta = beta # Annealing Curvature
        self.anneal_factor = 0

        self.alpha = self.anneal_factor**self.beta

        self.max_time = 0

        self.entropy = None
        self.track_entropy = False

    def sample_t(self, batch_size: int) -> torch.Tensor:

        log_times = self.log_mean_start + self.log_std_start*torch.randn((batch_size,))
        times_lognorm = torch.exp(log_times)
        times_lognorm = torch.clamp(times_lognorm, min=0.0, max=1.0)
        
        times_uniform = torch.rand((batch_size,))

        self.alpha = self.anneal_factor**self.beta
            
        times = (1-self.alpha)*times_lognorm + self.alpha*times_uniform

        counts, bin_edges = np.histogram(times, bins=20, density=False)
        self.max_time = max(bin_edges[:-1][counts > 10])    

        if(self.anneal_factor == 1):
            self.max_time = 1.0

        # self.entropy = self.get_entropy(times)

        return times
    
    def sample_iden_t(self, batch_size: int) -> torch.Tensor:

        log_time = self.log_mean_start + self.log_std_start*torch.randn([])
        time_lognorm = torch.exp(log_time)
        time_lognorm = torch.clamp(time_lognorm, min=0.0, max=1.0)

        time_uniform = torch.rand([])

        time = (1-(self.anneal_factor)**self.beta)*time_lognorm + (self.anneal_factor**self.beta)*time_uniform

        self.max_time = time

        if(self.anneal_factor == 1):
            self.max_time = 1.0

        # self.entropy = self.get_entropy(time)

        return torch.zeros(batch_size) + time

    @staticmethod
    def get_entropy(times: torch.Tensor, num_bins: int=20) -> float:
        counts, _ = np.histogram(times, bins=num_bins, density=False)
        probs = counts/counts.sum()

        return entropy(probs)
        
