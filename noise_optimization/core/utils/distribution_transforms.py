"""Distribution transforms for alternative latent space representations.

This module provides utilities for transforming between different latent distributions
while ensuring the final output to generative models is always N(0, I_D).

Key insight: The choice of "search space" (where we do optimization) can be different
from the "model space" (what the generative model expects). A good search space should:
1. Have uniform coverage of probability mass
2. Be bounded (better for GP kernels)
3. Preserve locality under the transform

Available transforms:
1. Hypercube: Sample uniform on [-1,1]^d, transform via inverse CDF to Gaussian
2. Hypersphere: Sample uniform on unit sphere with chi-distributed radius
3. Student-t: Sample from heavy-tailed t-distribution, transform to Gaussian
4. Sobol (quasi-random): Low-discrepancy sequences on [0,1]^d → Gaussian

Reference:
    - Input warping: Snoek et al. (2014). Input Warping for Bayesian Optimization.
"""
from __future__ import annotations

from typing import Optional, Tuple, Literal
from enum import Enum

import torch
import numpy as np
from torch.quasirandom import SobolEngine
from scipy import stats


class LatentDistribution(Enum):
    """Enumeration of supported latent space distributions."""
    GAUSSIAN = "gaussian"  # Standard N(0, I_d) - baseline
    HYPERCUBE = "hypercube"  # Uniform on [-1,1]^d → CDF transform
    HYPERSPHERE = "hypersphere"  # Uniform direction × chi-distributed radius
    STUDENT_T = "student_t"  # Heavy-tailed t-distribution → CDF transform
    SOBOL = "sobol"  # Low-discrepancy Sobol on [0,1]^d → CDF transform


class DistributionTransform:
    """Base class for distribution transforms.
    
    A transform maps from a "search space" distribution to a Gaussian distribution
    that can be fed to generative models.
    
    Attributes:
        dim: Dimensionality of the latent space
        device: Device for tensor operations
    """
    
    def __init__(self, dim: int, device: str = "cuda"):
        self.dim = dim
        self.device = device
    
    def sample(self, n_samples: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample from the search space distribution.
        
        Args:
            n_samples: Number of samples to generate
            generator: Optional PyTorch generator for reproducibility
            
        Returns:
            Samples in search space, shape (n_samples, dim)
        """
        raise NotImplementedError
    
    def to_gaussian(self, z: torch.Tensor) -> torch.Tensor:
        """Transform from search space to Gaussian.
        
        Args:
            z: Samples in search space, shape (n_samples, dim)
            
        Returns:
            Samples in Gaussian space N(0, I_d), shape (n_samples, dim)
        """
        raise NotImplementedError
    
    def from_gaussian(self, x: torch.Tensor) -> torch.Tensor:
        """Transform from Gaussian to search space (inverse transform).
        
        Args:
            x: Samples in Gaussian space, shape (n_samples, dim)
            
        Returns:
            Samples in search space, shape (n_samples, dim)
        """
        raise NotImplementedError
    
    def sample_gaussian(
        self,
        n_samples: int,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Sample from search space and transform to Gaussian.
        
        Convenience method that combines sample() and to_gaussian().
        
        Args:
            n_samples: Number of samples to generate
            generator: Optional PyTorch generator
            
        Returns:
            Gaussian samples, shape (n_samples, dim)
        """
        z = self.sample(n_samples, generator)
        return self.to_gaussian(z)
    
    @property
    def bounds(self) -> Optional[Tuple[float, float]]:
        """Return the bounds of the search space, or None if unbounded."""
        return None
    
    @property
    def name(self) -> str:
        """Return the name of this transform."""
        return self.__class__.__name__


class GaussianTransform(DistributionTransform):
    """Standard Gaussian distribution (baseline/identity transform).
    
    Search space: N(0, I_d)
    Model space: N(0, I_d)
    Transform: Identity
    """
    
    def sample(self, n_samples: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if generator is not None:
            return torch.randn(n_samples, self.dim, device=self.device, generator=generator)
        return torch.randn(n_samples, self.dim, device=self.device)
    
    def to_gaussian(self, z: torch.Tensor) -> torch.Tensor:
        return z  # Identity
    
    def from_gaussian(self, x: torch.Tensor) -> torch.Tensor:
        return x  # Identity
    
    @property
    def bounds(self) -> Optional[Tuple[float, float]]:
        return None  # Unbounded
    
    @property
    def name(self) -> str:
        return "gaussian"


class HypercubeTransform(DistributionTransform):
    """Uniform distribution on hypercube with CDF transform to Gaussian.
    
    Search space: Uniform on [-1, 1]^d (or [0, 1]^d)
    Model space: N(0, I_d)
    Transform: Φ^(-1)((z + 1) / 2) where Φ^(-1) is inverse standard normal CDF
    
    Advantages:
    - Bounded domain → well-defined trust regions
    - Uniform coverage → equal exploration everywhere
    - GP kernels work well on hypercubes
    - Euclidean distance ≈ probability distance
    """
    
    def __init__(
        self,
        dim: int,
        device: str = "cuda",
        range_min: float = -1.0,
        range_max: float = 1.0,
    ):
        super().__init__(dim, device)
        self.range_min = range_min
        self.range_max = range_max
    
    def sample(self, n_samples: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if generator is not None:
            u = torch.rand(n_samples, self.dim, device=self.device, generator=generator)
        else:
            u = torch.rand(n_samples, self.dim, device=self.device)
        return self.range_min + (self.range_max - self.range_min) * u
    
    def to_gaussian(self, z: torch.Tensor) -> torch.Tensor:
        # Map from [range_min, range_max] to [0, 1]
        u = (z - self.range_min) / (self.range_max - self.range_min)
        # Clamp to avoid numerical issues at boundaries
        u = torch.clamp(u, 1e-7, 1 - 1e-7)
        # Apply inverse CDF: Φ^(-1)(u) = √2 * erfinv(2u - 1)
        return torch.sqrt(torch.tensor(2.0, device=z.device)) * torch.special.erfinv(2 * u - 1)
    
    def from_gaussian(self, x: torch.Tensor) -> torch.Tensor:
        # Apply CDF: Φ(x) = 0.5 * (1 + erf(x / √2))
        u = 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0, device=x.device))))
        # Map from [0, 1] to [range_min, range_max]
        return self.range_min + (self.range_max - self.range_min) * u
    
    @property
    def bounds(self) -> Tuple[float, float]:
        return (self.range_min, self.range_max)
    
    @property
    def name(self) -> str:
        return "hypercube"


class HypersphereTransform(DistributionTransform):
    """Spherical decomposition: direction × radius.
    
    Search space: (direction on S^{d-1}, radius ~ χ_d)
    Model space: N(0, I_d)
    
    A Gaussian in d dimensions can be written as:
        x = r * u  where:
        - u is uniform on the unit sphere S^{d-1}
        - r follows a chi distribution with d degrees of freedom
    
    This decomposition allows separate optimization of magnitude vs direction.
    """
    
    def sample(self, n_samples: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # Sample direction: standard Gaussian normalized to unit sphere
        if generator is not None:
            direction = torch.randn(n_samples, self.dim, device=self.device, generator=generator)
        else:
            direction = torch.randn(n_samples, self.dim, device=self.device)
        
        # Normalize to unit sphere
        direction = direction / (direction.norm(dim=1, keepdim=True) + 1e-12)
        
        # Sample radius from chi distribution (sqrt of chi-squared with d degrees of freedom)
        # For uniform volume in sphere: r ~ U[0,1]^(1/d) * R_max
        # But for matching Gaussian, we need chi_d distribution
        if generator is not None:
            # Chi distribution: sqrt of sum of squared Gaussians
            chi_samples = torch.randn(n_samples, self.dim, device=self.device, generator=generator)
        else:
            chi_samples = torch.randn(n_samples, self.dim, device=self.device)
        radius = chi_samples.norm(dim=1, keepdim=True)
        
        # Return as (direction, radius) encoded in the last dimension
        # We'll represent this as [direction * (some_max_radius), radius / max_radius]
        # For simplicity, just return the full Gaussian sample for now
        return direction * radius
    
    def to_gaussian(self, z: torch.Tensor) -> torch.Tensor:
        # Already Gaussian if sampled correctly
        return z
    
    def from_gaussian(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    @property
    def bounds(self) -> Optional[Tuple[float, float]]:
        return None  # Unbounded in Cartesian, but constrained to sphere
    
    @property
    def name(self) -> str:
        return "hypersphere"


class SphericalSearchTransform(DistributionTransform):
    """Spherical search space with explicit radius and angle representation.
    
    This is different from HypersphereTransform - here we explicitly work in
    spherical coordinates for search, which may have better properties for
    some optimization algorithms.
    
    Search space: 
        - radius: [0, max_radius] (or transformed to [0, 1])
        - angles: hyperspherical angles (d-1 angles for d dimensions)
    
    Model space: N(0, I_d)
    """
    
    def __init__(
        self,
        dim: int,
        device: str = "cuda",
        max_radius: float = 4.0,  # ~99.99% of Gaussian mass within this radius
    ):
        super().__init__(dim, device)
        self.max_radius = max_radius
    
    def sample(self, n_samples: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # For now, sample Gaussian and convert to spherical internally
        # This gives us the right distribution
        if generator is not None:
            x = torch.randn(n_samples, self.dim, device=self.device, generator=generator)
        else:
            x = torch.randn(n_samples, self.dim, device=self.device)
        return x
    
    def to_gaussian(self, z: torch.Tensor) -> torch.Tensor:
        return z
    
    def from_gaussian(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    @property
    def name(self) -> str:
        return "spherical_search"


class StudentTTransform(DistributionTransform):
    """Student-t distribution with CDF transform to Gaussian.
    
    Search space: t_ν(0, I_d) with ν degrees of freedom
    Model space: N(0, I_d)
    Transform: Φ^(-1)(F_t(z)) where F_t is t-distribution CDF
    
    Advantages:
    - Heavy tails → more exploration in extreme regions
    - Robust to outliers
    - ν controls tail heaviness (lower ν = heavier tails)
    
    Note: Requires scipy for t-distribution CDF/inverse CDF.
    """
    
    def __init__(
        self,
        dim: int,
        device: str = "cuda",
        df: float = 3.0,  # Degrees of freedom (3-5 typical)
    ):
        super().__init__(dim, device)
        self.df = df
    
    def sample(self, n_samples: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # Sample from Student-t using PyTorch
        # t = N(0,1) / sqrt(χ²_ν / ν)
        if generator is not None:
            normal = torch.randn(n_samples, self.dim, device=self.device, generator=generator)
            chi2 = torch.randn(n_samples, 1, device=self.device, generator=generator) ** 2
            for _ in range(int(self.df) - 1):
                chi2 = chi2 + torch.randn(n_samples, 1, device=self.device, generator=generator) ** 2
        else:
            normal = torch.randn(n_samples, self.dim, device=self.device)
            chi2 = torch.randn(n_samples, 1, device=self.device) ** 2
            for _ in range(int(self.df) - 1):
                chi2 = chi2 + torch.randn(n_samples, 1, device=self.device) ** 2
        
        t_samples = normal / torch.sqrt(chi2 / self.df + 1e-12)
        return t_samples
    
    def to_gaussian(self, z: torch.Tensor) -> torch.Tensor:
        # Transform via CDF: Φ^(-1)(F_t(z))
        # Use scipy for numerical stability
        z_np = z.detach().cpu().numpy()
        
        # Apply t-CDF element-wise
        u = stats.t.cdf(z_np, df=self.df)
        u = np.clip(u, 1e-7, 1 - 1e-7)
        
        # Apply inverse Gaussian CDF
        x_np = stats.norm.ppf(u)
        
        return torch.tensor(x_np, dtype=z.dtype, device=z.device)
    
    def from_gaussian(self, x: torch.Tensor) -> torch.Tensor:
        # Inverse transform: F_t^(-1)(Φ(x))
        x_np = x.detach().cpu().numpy()
        
        # Apply Gaussian CDF
        u = stats.norm.cdf(x_np)
        u = np.clip(u, 1e-7, 1 - 1e-7)
        
        # Apply inverse t-CDF
        z_np = stats.t.ppf(u, df=self.df)
        
        return torch.tensor(z_np, dtype=x.dtype, device=x.device)
    
    @property
    def bounds(self) -> Optional[Tuple[float, float]]:
        return None  # Unbounded but heavy-tailed
    
    @property
    def name(self) -> str:
        return f"student_t_df{self.df}"


class SobolTransform(DistributionTransform):
    """Low-discrepancy Sobol sequence with CDF transform to Gaussian.
    
    Search space: Sobol sequence on [0, 1]^d (quasi-random, low-discrepancy)
    Model space: N(0, I_d)
    Transform: Φ^(-1)(u) where u is Sobol sample
    
    Advantages:
    - Better space-filling than random sampling
    - Provably lower discrepancy
    - Reduced variance in Monte Carlo estimates
    - Deterministic (reproducible)
    """
    
    def __init__(
        self,
        dim: int,
        device: str = "cuda",
        seed: int = 42,
        scramble: bool = True,
    ):
        super().__init__(dim, device)
        self.seed = seed
        self.scramble = scramble
        self._engine = SobolEngine(dim, scramble=scramble, seed=seed)
        self._sample_count = 0
    
    def sample(self, n_samples: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # Sobol is deterministic, generator is ignored
        u = self._engine.draw(n_samples).to(dtype=torch.float32, device=self.device)
        self._sample_count += n_samples
        return u
    
    def to_gaussian(self, z: torch.Tensor) -> torch.Tensor:
        # z is in [0, 1]^d, transform to Gaussian via inverse CDF
        u = torch.clamp(z, 1e-7, 1 - 1e-7)
        return torch.sqrt(torch.tensor(2.0, device=z.device)) * torch.special.erfinv(2 * u - 1)
    
    def from_gaussian(self, x: torch.Tensor) -> torch.Tensor:
        # Transform Gaussian to [0, 1] via CDF
        u = 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0, device=x.device))))
        return u
    
    def reset(self, seed: Optional[int] = None):
        """Reset the Sobol engine to restart the sequence."""
        if seed is not None:
            self.seed = seed
        self._engine = SobolEngine(self.dim, scramble=self.scramble, seed=self.seed)
        self._sample_count = 0
    
    @property
    def bounds(self) -> Tuple[float, float]:
        return (0.0, 1.0)
    
    @property
    def name(self) -> str:
        return "sobol"


def get_transform(
    distribution: str,
    dim: int,
    device: str = "cuda",
    **kwargs
) -> DistributionTransform:
    """Factory function to get a distribution transform by name.
    
    Args:
        distribution: Name of the distribution (gaussian, hypercube, hypersphere, student_t, sobol)
        dim: Dimensionality of the latent space
        device: Device for tensor operations
        **kwargs: Additional arguments for specific transforms
        
    Returns:
        DistributionTransform instance
    """
    dist_lower = distribution.lower().strip()
    
    if dist_lower in ("gaussian", "normal", "standard"):
        return GaussianTransform(dim, device)
    elif dist_lower in ("hypercube", "cube", "uniform"):
        return HypercubeTransform(
            dim, device,
            range_min=kwargs.get("range_min", -1.0),
            range_max=kwargs.get("range_max", 1.0),
        )
    elif dist_lower in ("hypersphere", "sphere", "spherical"):
        return HypersphereTransform(dim, device)
    elif dist_lower in ("student_t", "student", "t"):
        return StudentTTransform(
            dim, device,
            df=kwargs.get("df", 3.0),
        )
    elif dist_lower in ("sobol", "quasi", "low_discrepancy"):
        return SobolTransform(
            dim, device,
            seed=kwargs.get("seed", 42),
            scramble=kwargs.get("scramble", True),
        )
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def verify_gaussian_output(
    transform: DistributionTransform,
    n_samples: int = 10000,
    significance: float = 0.05,
) -> dict:
    """Verify that a transform produces approximately Gaussian output.
    
    Performs statistical tests on the transformed samples to verify:
    1. Mean ≈ 0
    2. Variance ≈ 1
    3. Distribution is approximately Gaussian (per-dimension KS test)
    
    Args:
        transform: Distribution transform to test
        n_samples: Number of samples to generate
        significance: Significance level for statistical tests
        
    Returns:
        Dictionary with test results
    """
    # Generate samples in search space
    z = transform.sample(n_samples)
    
    # Transform to Gaussian
    x = transform.to_gaussian(z)
    
    # Compute statistics
    mean = x.mean(dim=0)
    std = x.std(dim=0)
    
    # Per-dimension Kolmogorov-Smirnov test
    x_np = x.detach().cpu().numpy()
    ks_stats = []
    ks_pvals = []
    for d in range(min(transform.dim, 10)):  # Test first 10 dimensions
        ks_stat, ks_pval = stats.kstest(x_np[:, d], 'norm')
        ks_stats.append(ks_stat)
        ks_pvals.append(ks_pval)
    
    # Summary statistics
    mean_error = float((mean ** 2).mean().sqrt().item())
    std_error = float(((std - 1) ** 2).mean().sqrt().item())
    
    # Check if tests pass
    mean_pass = mean_error < 0.1  # Mean within 0.1 of 0
    std_pass = std_error < 0.1   # Std within 0.1 of 1
    ks_pass = all(p > significance for p in ks_pvals)  # All dimensions pass KS test
    
    return {
        "transform": transform.name,
        "n_samples": n_samples,
        "dim": transform.dim,
        "mean_error": mean_error,
        "std_error": std_error,
        "mean_pass": mean_pass,
        "std_pass": std_pass,
        "ks_stats": ks_stats,
        "ks_pvals": ks_pvals,
        "ks_pass": ks_pass,
        "all_pass": mean_pass and std_pass and ks_pass,
    }

