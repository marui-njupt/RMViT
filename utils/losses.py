"""Stage-specific reconstruction losses with explicit weights and normalization."""
import torch


def stage1_loss(pred, target, gradient_weight, *, normalization_divisor):
    p, y = pred / normalization_divisor, target / normalization_divisor
    mse = (p - y).square().mean()
    dx = (p[..., 1:] - p[..., :-1]) - (y[..., 1:] - y[..., :-1])
    dy = (p[..., 1:, :] - p[..., :-1, :]) - (y[..., 1:, :] - y[..., :-1, :])
    grad = dx.abs().mean() + dy.abs().mean()

    return mse + gradient_weight * grad, {'mse': mse, 'gradient': grad}


def stage2_loss(pred, target, sparse_rm, global_weight, sampled_weight, *, normalization_divisor, observation_epsilon):
    p, y, rm = pred / normalization_divisor, target / normalization_divisor, sparse_rm / normalization_divisor
    err = (p - y).square()
    mse = err.mean()


    # Preserve the original convention: zero-valued observations are excluded.
    mask = (rm.abs() > observation_epsilon).float()
    sampled = ((err * mask).sum((1, 2, 3)) /
               (mask.sum((1, 2, 3)) + observation_epsilon)).mean()
    return global_weight * mse + sampled_weight * sampled, {'mse': mse, 'sampled_mse': sampled}
