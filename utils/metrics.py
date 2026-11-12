"""Full-grid metrics without prediction clipping or conversion to physical units."""
import math

import numpy as np
from skimage.metrics import structural_similarity


def sample_metrics(pred, target, ssim_options, *, normalization_divisor):
    pred = np.asarray(pred, dtype=np.float64) / normalization_divisor
    target = np.asarray(target, dtype=np.float64) / normalization_divisor
    if pred.shape != (256, 256) or target.shape != pred.shape:
        raise ValueError('Metric domain must be one full 256x256 map.')
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise FloatingPointError('Nonfinite map; do not silently omit a sample.')
    sse = float(np.square(pred - target).sum())
    energy = float(np.square(target).sum())
    return {'sse': sse, 'target_energy': energy, 'pixels': target.size,
            'rmse': math.sqrt(sse / target.size),
            'nmse': sse / energy if energy > 0 else None,
            'ssim': float(structural_similarity(target, pred, **ssim_options))}


class MetricAccumulator:
    def __init__(self):
        self.sse = self.energy = self.ssim_sum = 0.0
        self.pixels = self.samples = 0

    def add(self, row):
        self.sse += row['sse']
        self.energy += row['target_energy']
        self.pixels += row['pixels']
        self.ssim_sum += row['ssim']
        self.samples += 1

    def result(self):
        if self.samples == 0:
            raise ValueError('Cannot report empty/partial results as a complete test.')
        return {'samples': self.samples, 'pixels': self.pixels,
                'sse': self.sse, 'target_energy': self.energy,
                'rmse': math.sqrt(self.sse / self.pixels),
                'nmse': self.sse / self.energy if self.energy > 0 else None,
                'ssim': self.ssim_sum / self.samples}
