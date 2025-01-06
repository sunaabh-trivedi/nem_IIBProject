import torch

def kde(data: torch.Tensor, query: torch.Tensor, bandwidth: float = 1.0):
    # buffer: Tensor of shape [N, d]
    # points: Tensor of shape [M, d] (query points where density is evaluated)
    # bandwidth: scalar for the Gaussian kernel bandwidth

    N, d = data.shape
    M, _ = query.shape
    
    # Normalization factor (for Gaussian KDE)
    normalization = (2 * torch.pi * bandwidth**2) ** (d / 2)

    # densities = unnormalize_kde(data, query, bandwidth) / normalization
    densities = log_unnormalize_kde(data, query, bandwidth).exp() / normalization
    return densities


def unnormalize_kde(data: torch.Tensor, query: torch.Tensor, bandwidth: float = 1.0):
    # buffer: Tensor of shape [N, d]
    # points: Tensor of shape [M, d] (query points where density is evaluated)
    # bandwidth: scalar for the Gaussian kernel bandwidth

    N, d = data.shape
    M, _ = query.shape
    
    # Calculate squared differences between buffer and points
    diffs = data.unsqueeze(0) - query.unsqueeze(1)  # Shape: [M, N, d]
    squared_diffs = diffs.pow(2).sum(dim=-1)  # Shape: [M, N]

    # Gaussian kernel
    kernel_vals = torch.exp(-squared_diffs / (2 * bandwidth**2))

    # Estimate density
    unnormalized_densities = kernel_vals.mean(dim=1)  # Shape: [M]
    
    return unnormalized_densities


def log_unnormalize_kde(data: torch.Tensor, query: torch.Tensor, bandwidth: float = 1.0):
    # buffer: Tensor of shape [N, d]
    # points: Tensor of shape [M, d] (query points where density is evaluated)
    # bandwidth: scalar for the Gaussian kernel bandwidth

    N, _ = data.shape
    M, _ = query.shape
    
    # Calculate squared differences between buffer and points
    diffs = data.unsqueeze(0) - query.unsqueeze(1)  # Shape: [M, N, d]
    squared_diffs = diffs.pow(2).sum(dim=-1)  # Shape: [M, N]
    
    # Compute log kernel values in a numerically stable way
    log_kernel_vals = -squared_diffs / (2 * bandwidth**2)
    
    # Use logsumexp for numerically stable summation in log space
    log_unnormalized_density = torch.logsumexp(log_kernel_vals, dim=1)  # Shape: [M]
    
    return log_unnormalized_density