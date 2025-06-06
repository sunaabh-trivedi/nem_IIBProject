from typing import Callable, Iterable, NamedTuple, Tuple

import torch


class AISData(NamedTuple):
    """Log weights and samples generated by annealed importance sampling."""

    x: torch.Tensor
    log_w: torch.Tensor
    add_count: torch.Tensor
    log_q: torch.Tensor


class ReplayBuffer:
    def __init__(
        self,
        dim: int,
        max_length: int,
        min_sample_length: int,
        initial_sampler: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
        device: str = "cpu",
        temperature: float = 1.0,
        with_q=False,
    ):
        """
        Create replay buffer for batched sampling and adding of data.
        Args:
            dim: dimension of x data
            max_length: maximum length of the buffer
            min_sample_length: minimum length of buffer required for sampling
            initial_sampler: sampler producing x and log_w, used to fill the buffer up to
                the min sample length. The initialised flow + AIS may be used here,
                or we may desire to use AIS with more distributions to give the flow a "good start".
            device: replay buffer device
            temperature: rate at which we anneal the sampling probability of experience as new batches get added
                anneal_temperature of 0 gives uniform sampling

        The `max_length` and `min_sample_length` should be sufficiently long to prevent overfitting
        to the replay data. For example, if `min_sample_length` is equal to the
        sampling batch size, then we may overfit to the first batch of data, as we would update
        on it many times during the start of training.
        """
        assert min_sample_length < max_length
        self.dim = dim
        self.max_length = max_length
        self.min_sample_length = min_sample_length
        self.buffer = AISData(
            x=torch.zeros(self.max_length, dim).to(device),
            log_w=torch.zeros(
                self.max_length,
            ).to(device),
            add_count=torch.zeros(
                self.max_length,
            ).to(device),
            log_q=torch.zeros(self.max_length).to(device) if with_q else None
        )
        self.possible_indices = torch.arange(self.max_length).to(device)
        self.device = device
        self.current_index = 0
        self.current_add_count = 0
        self.is_full = False  # whether the buffer is full
        self.can_sample = False  # whether the buffer is full enough to begin sampling
        self.temperature = temperature
        self.with_q = with_q
        self.current_add_count = 1

    @torch.no_grad()
    def add(self, x: torch.Tensor, log_w: torch.Tensor, log_q=None):
        """Add a batch of generated data to the replay buffer."""
        batch_size = x.shape[0]
        x = x.to(self.device)
        log_w = log_w.to(self.device)
        log_q = log_q.to(self.device)
        indices = (torch.arange(batch_size) + self.current_index).to(self.device) % self.max_length
        self.buffer.x[indices] = x
        self.buffer.log_w[indices] = log_w
        if self.with_q:
            self.buffer.log_q[indices] = log_q
        self.buffer.add_count[indices] = self.current_add_count
        new_index = self.current_index + batch_size
        if not self.is_full:
            self.is_full = new_index >= self.max_length
            self.can_sample = new_index >= self.min_sample_length
        self.current_index = new_index % self.max_length
        self.current_add_count += 1

    def get_last_n_inserted(self, num_to_get: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.is_full:
            assert num_to_get <= self.max_length
        else:
            assert num_to_get < self.current_index

        start_idx = self.current_index - num_to_get
        idxs = [torch.arange(max(start_idx, 0), self.current_index)]
        if start_idx < 0:
            idxs.append(torch.arange(self.max_length + start_idx, self.max_length))

        idx = torch.cat(idxs)
        if self.with_q:
            return self.buffer.x[idx], self.buffer.log_w[idx], self.buffer.log_q[idx]
        else:
            return self.buffer.x[idx], self.buffer.log_w[idx]
        
    @torch.no_grad()
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a batch of sampled data, if the batch size is specified then the batch will have
        a leading axis of length batch_size, otherwise the default self.batch_size will be used."""
        if not self.can_sample:
            raise Exception("Buffer must be at minimum length before calling sample")
        max_index = self.max_length if self.is_full else self.current_index
        log_probs = torch.pow(self.buffer.log_w, self.temperature)
        probs = torch.exp(torch.clamp(log_probs, max=20))
        indices = torch.multinomial(probs, num_samples=batch_size, replacement=False).to(
        self.device
        )  # sample uniformly
        if self.with_q:
            return self.buffer.x[indices], self.buffer.log_w[indices], self.buffer.log_q[indices], indices
        else:
            return self.buffer.x[indices], self.buffer.log_w[indices]

    def sample_n_batches(
        self, batch_size: int, n_batches: int
    ) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        """Returns a list of batches."""
        if self.with_q:
            x, log_w, log_q = self.sample(batch_size * n_batches)
        else:
            x, log_w = self.sample(batch_size * n_batches)
        x_batches = torch.chunk(x, n_batches)
        log_w_batches = torch.chunk(log_w, n_batches)
        if self.with_q:
            log_q_batches = torch.chunk(log_q, n_batches)
            dataset = [(x, log_w, log_q) for x, log_w, log_q in zip(x_batches, log_w_batches, log_q_batches)]
        else:
            dataset = [(x, log_w) for x, log_w in zip(x_batches, log_w_batches)]
        return dataset
    
    def __len__(self):
        return len(self.buffer.x)


if __name__ == "__main__":
    # to check that the replay buffer runs
    dim = 5
    batch_size = 3
    n_batches_total_length = 2
    length = n_batches_total_length * batch_size
    min_sample_length = int(length * 0.5)

    def initial_sampler():
        return (torch.ones(batch_size, dim), torch.zeros(batch_size))

    buffer = ReplayBuffer(dim, length, min_sample_length, initial_sampler, temperature=0.0)
    n_batches = 3
    for i in range(100):
        buffer.add(torch.ones(batch_size, dim), torch.zeros(batch_size))
        batch = buffer.sample(batch_size)
