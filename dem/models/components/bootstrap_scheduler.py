import torch
from dem.models.components.noise_schedules import *
from dem.models.components.annealing_schedule import *

class BootstrapSchedule:
    """
    The BootstrapSchedule is iterated during a Bootstrapping-sampler training, following the pseudo-code below:
            bootstrap_scheduler = BootstrapSchedule(num_steps)
            for t_last, t in bootstrap_scheduler:
                x_t ~ q(x_t | x_0)
                predictor = net(x_t, t)
                bootstrap_estimator = f(x_t, t_last, net)
                loss = loss_fn(predictor, bootstrap_estimator)
    """
    def __init__(self):
        self.time_splits = None
        self.index = 1
        self.max_tries = 50

    def initialise_time_splits(self, annealing_schedule=None):
        """
        initialise the time splits lazily once all necessary information is available.
        """
        self.time_splits = self.time_spliter(annealing_schedule)
        self.time_splits = torch.cat((torch.tensor([0.]), self.time_splits))  # Pad a 0 at the beginning

    def time_spliter(self, annealing_schedule=None) -> torch.Tensor:
        """
        Split [0, 1] into sub-intervals.
        Must be implemented in derived classes.
        """
        raise NotImplementedError

    def __next__(self):
        if self.index < len(self.time_splits):
            res = self.time_splits[:self.index + 1]
            self.index += 1
            return res
        else:
            raise StopIteration

    def t_to_index(self, t: torch.Tensor) -> torch.Tensor:
        indexes = []
        for t_ in t:
            indexes.append(torch.sum(self.time_splits <= t_).item() - 1)
        return torch.tensor(indexes, dtype=torch.long).to(t.device)  

    def index_to_t(self, index: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.time_splits[index], self.time_splits[index + 1]])  # [t_last, t_current]

    def sample_t(self, index: torch.Tensor) -> torch.Tensor:
        return torch.rand_like(index.float()) * (self.time_splits[index + 1] - self.time_splits[index]) + self.time_splits[index]

    def __iter__(self):
        self.index = 1
        return self

    def __len__(self):
        return len(self.time_splits) - 1


class LinearBootstrapSchedule(BootstrapSchedule):
    def __init__(self, noise_scheduler: GeometricNoiseSchedule, num_bootstrap_steps: int):
        super().__init__()
        self.num_bootstrap_steps = num_bootstrap_steps
        self.initialise_time_splits()

    def time_spliter(self, annealing_schedule=None) -> torch.Tensor:
        return torch.linspace(0, 1, self.num_bootstrap_steps + 1)


class GeometricBootstrapSchedule(BootstrapSchedule):
    def __init__(self, noise_scheduler: GeometricNoiseSchedule, variance: float):
        super().__init__()
        self.h = noise_scheduler.h
        self.h_series = [self.h(t) for t in torch.linspace(0, 1, steps=1000)]
        self.variance = variance
        self.initialise_time_splits()

    def time_spliter(self, annealing_schedule=None) -> torch.Tensor:
        k = torch.round((self.h_series[-1] - self.h_series[0]) / self.variance)
        k = int(k.item())
        t = [0.]
        h_idx, sigma_c, sigma_l = 0, self.h_series[0], self.h_series[0]
        end_flag = False
        for i in range(1, k):
            while (sigma_l - sigma_c) < self.variance:
                sigma_l = self.h_series[h_idx]
                h_idx += 1
                if h_idx == 999:
                    end_flag = True
                    break
            sigma_c = sigma_l
            t.append(h_idx / 1000)
            if end_flag:
                break
        t.append(1.0)
        print("checky t::", t)
        return torch.tensor(t)


class AnnealingBootstrapSchedule(BootstrapSchedule):
    def __init__(self, noise_scheduler: GeometricNoiseSchedule, variance: float,
                 sample_size: int = 10000, anneal_factor_step: float = 0.05):
        super().__init__()
        self.noise_scheduler = noise_scheduler
        self.variance = variance
        self.sample_size = sample_size
        self.anneal_factor_step = anneal_factor_step
        self.last_anneal_factor = -1.0

    def time_spliter(self, annealing_schedule: AnnealingSchedule) -> torch.Tensor:
        current_anneal_factor = annealing_schedule.anneal_factor
        # If time_splits is None or the factor changed enough, recompute the time splits
        if (self.time_splits is None or
            abs(current_anneal_factor - self.last_anneal_factor) >= self.anneal_factor_step):
            splits = self._compute_time_splits(annealing_schedule)
            # Ensure 0.0 at front, 1.0 at end
            if splits[0] > 1e-9:
                splits = torch.cat([torch.tensor([0.0], device=splits.device), splits])
            if splits[-1] < 0.9999999:
                splits = torch.cat([splits, torch.tensor([1.0], device=splits.device)])
            self.time_splits = splits
            self.last_anneal_factor = current_anneal_factor

        return self.time_splits

    def _compute_time_splits(self, annealing_schedule: AnnealingSchedule) -> torch.Tensor:
        T = annealing_schedule.sample_t(self.sample_size)
        H = self.noise_scheduler.h(T)

        sorted_indices = torch.argsort(T)
        T_sorted = T[sorted_indices]
        H_sorted = H[sorted_indices]

        sample_range = T_sorted.max() - T_sorted.min()
        normalized_range = sample_range / 1.0
        var = 0.01 + (0.1 - 0.01) * normalized_range

        splits = [T_sorted[0].item()]
        current_h = H_sorted[0].item()

        for i in range(1, len(H_sorted)):
            if H_sorted[i].item() - current_h >= var:
                splits.append(T_sorted[i].item())
                current_h = H_sorted[i].item()

        return torch.tensor(splits, device=T.device)

    def sample_t(self, index: torch.Tensor) -> torch.Tensor:
        index = torch.clamp(index, 0, len(self.time_splits) - 2)
        lower_bound = self.time_splits[index]
        upper_bound = self.time_splits[index + 1]
        return torch.rand_like(index.float()) * (upper_bound - lower_bound) + lower_bound


