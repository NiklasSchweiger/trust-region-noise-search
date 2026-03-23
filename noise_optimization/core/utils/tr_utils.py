import torch
import numpy as np
from typing import Tuple, List, Optional, Callable, Dict, Any
from torch.quasirandom import SobolEngine


# Global cache for SobolEngines to avoid heavy re-initialization in high dimensions
# Key: (dim, seed, scramble)
_SOBOL_CACHE: Dict[Tuple[int, int, bool], SobolEngine] = {}


def get_sobol_engine(dim: int, scramble: bool, seed: int) -> SobolEngine:
    """Get a cached SobolEngine or create a new one."""
    key = (dim, seed, scramble)
    if key not in _SOBOL_CACHE:
        _SOBOL_CACHE[key] = SobolEngine(dim, scramble=scramble, seed=seed)
    return _SOBOL_CACHE[key]


# ==============================================================================
# CENTER SELECTION HELPERS
# ==============================================================================


def select_centers_diverse(
    Z_archive: torch.Tensor,
    R_archive: torch.Tensor,
    n_regions: int,
    min_dist: float,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select high-performing but spatially diverse trust-region centers."""
    if Z_archive.shape[0] == 0:
        return (
            torch.zeros(0, Z_archive.shape[1], device=device),
            torch.zeros(0, device=device),
        )

    vals = R_archive.view(-1)
    sorted_idx = torch.argsort(vals, descending=True)

    selected_indices: List[torch.Tensor] = []
    selected_z_list: List[torch.Tensor] = []

    if len(sorted_idx) > 0:
        best_idx = sorted_idx[0]
        selected_indices.append(best_idx)
        selected_z_list.append(Z_archive[best_idx])

    for idx in sorted_idx[1:]:
        if len(selected_indices) >= n_regions:
            break

        candidate_z = Z_archive[idx]

        is_far_enough = True
        if len(selected_z_list) > 0:
            current_centers = torch.stack(selected_z_list)
            dists = torch.norm(current_centers - candidate_z.unsqueeze(0), dim=1)
            if torch.min(dists) < min_dist:
                is_far_enough = False

        if is_far_enough:
            selected_indices.append(idx)
            selected_z_list.append(candidate_z)

    if len(selected_indices) < n_regions:
        used_set = {i.item() for i in selected_indices}
        for idx in sorted_idx:
            if len(selected_indices) >= n_regions:
                break
            if idx.item() not in used_set:
                selected_indices.append(idx)
                used_set.add(idx.item())

    final_indices = torch.stack(selected_indices)
    centers = Z_archive[final_indices]
    values = vals[final_indices]
    return centers, values


def select_centers_clustering(
    Z_archive: torch.Tensor,
    R_archive: torch.Tensor,
    n_regions: int,
    top_percentile: float = 0.2,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select centers by clustering top-performing archive points."""
    N = Z_archive.shape[0]
    if N < n_regions:
        return select_centers_diverse(Z_archive, R_archive, n_regions, min_dist=0.0, device=device)

    vals = R_archive.view(-1)
    k_elite = max(n_regions, int(N * top_percentile))
    _, top_idxs = torch.topk(vals, k=k_elite)

    Z_elite = Z_archive[top_idxs]
    R_elite = vals[top_idxs]

    try:
        centroids = Z_elite[:n_regions].clone()

        for _ in range(5):
            dists = torch.cdist(Z_elite, centroids)
            cluster_ids = torch.argmin(dists, dim=1)

            new_centroids: List[torch.Tensor] = []
            for i in range(n_regions):
                mask = cluster_ids == i
                if mask.sum() > 0:
                    new_centroids.append(Z_elite[mask].mean(dim=0))
                else:
                    new_centroids.append(centroids[i])
            centroids = torch.stack(new_centroids)

        dists = torch.cdist(Z_elite, centroids)
        cluster_ids = torch.argmin(dists, dim=1)

        final_centers_list: List[torch.Tensor] = []
        final_values_list: List[torch.Tensor] = []

        for i in range(n_regions):
            mask = cluster_ids == i
            if mask.sum() > 0:
                cluster_indices_local = torch.nonzero(mask).view(-1)
                cluster_rewards = R_elite[cluster_indices_local]
                best_in_cluster_idx = torch.argmax(cluster_rewards)
                global_idx_in_elite = cluster_indices_local[best_in_cluster_idx]

                final_centers_list.append(Z_elite[global_idx_in_elite])
                final_values_list.append(R_elite[global_idx_in_elite])

        if len(final_centers_list) == 0:
            return select_centers_diverse(Z_archive, R_archive, n_regions, min_dist=0.0, device=device)

        centers = torch.stack(final_centers_list)
        values = torch.stack(final_values_list)
        return centers, values

    except Exception as e:
        print(f"Clustering selection failed: {e}, falling back to diverse.")
        return select_centers_diverse(Z_archive, R_archive, n_regions, min_dist=0.0, device=device)


# ==============================================================================
# LENGTH UPDATE HELPERS
# ==============================================================================


def calculate_1_5_rule_length(
    current_length: float,
    success_rate: float,
    min_length: float,
    max_length: float,
    target_rate: float = 0.2,
    growth_factor: float = 1.5,
    shrink_factor: float = 0.5,
) -> float:
    """Adjust trust-region length using the 1/5 success rule."""
    if success_rate > target_rate:
        new_len = current_length * growth_factor
    elif success_rate < target_rate:
        new_len = current_length * shrink_factor
    else:
        new_len = current_length
    return max(min_length, min(max_length, new_len))


def calculate_variance_based_length(
    current_length: float,
    batch_rewards: torch.Tensor,
    min_length: float,
    max_length: float,
    std_threshold_low: float = 0.01,
    std_threshold_high: float = 0.5,
) -> float:
    """Adjust length based on reward variance in the batch."""
    std = float(batch_rewards.std().item()) if batch_rewards.numel() > 1 else 0.0
    if std < std_threshold_low:
        return max(min_length, min(max_length, current_length * 1.5))
    if std > std_threshold_high:
        return max(min_length, min(max_length, current_length * 0.75))
    return current_length


# ==============================================================================
# SCORING FUNCTIONS
# ==============================================================================


def score_candidates_random(candidates: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Score candidates using random values (surrogate-free)."""
    if generator is not None:
        return torch.randn(
            candidates.shape[0],
            device=candidates.device,
            dtype=torch.float32,
            generator=generator,
        )
    return torch.randn(candidates.shape[0], device=candidates.device, dtype=torch.float32)


def score_candidates_linear(
    candidates: torch.Tensor,
    z_center: torch.Tensor,
    Z_archive: torch.Tensor,
    R_archive: torch.Tensor,
    n_neighbors: int = 20,
    regularization: float = 1e-4,
) -> torch.Tensor:
    """Score using local linear gradient approximation (Ridge Regression)."""
    d = z_center.shape[0]
    device = candidates.device

    if Z_archive is None or Z_archive.size(0) < d + 2:
        return score_candidates_random(candidates)

    dists = torch.cdist(z_center.unsqueeze(0), Z_archive)
    k = min(Z_archive.size(0), max(n_neighbors, d + 2))

    _, idxs = torch.topk(dists.view(-1), k=k, largest=False)

    Z_neigh = Z_archive[idxs]
    R_neigh = R_archive[idxs]

    z_mean = Z_neigh.mean(dim=0)
    z_std = Z_neigh.std(dim=0).clamp_min(1e-6)
    r_mean = R_neigh.mean()
    r_std = R_neigh.std().clamp_min(1e-6)

    Z_norm = (Z_neigh - z_mean) / z_std
    R_norm = (R_neigh - r_mean) / r_std

    Z_bias = torch.cat([Z_norm, torch.ones(k, 1, device=device)], dim=1)
    I = torch.eye(d + 1, device=device) * regularization

    try:
        w = torch.linalg.solve(Z_bias.T @ Z_bias + I, Z_bias.T @ R_norm)
        C_norm = (candidates - z_mean) / z_std
        C_bias = torch.cat([C_norm, torch.ones(candidates.shape[0], 1, device=device)], dim=1)
        scores = C_bias @ w
        return scores.view(-1)
    except RuntimeError:
        return score_candidates_random(candidates)


def score_candidates_momentum(
    candidates: torch.Tensor,
    z_center: torch.Tensor,
    prev_center: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Score using momentum / directional consistency."""
    if prev_center is None:
        return score_candidates_random(candidates)

    direction = z_center - prev_center
    norm = direction.norm()
    if norm < 1e-6:
        return score_candidates_random(candidates)

    direction = direction / norm
    cand_vecs = candidates - z_center.unsqueeze(0)
    scores = cand_vecs @ direction
    return scores


def score_candidates_idw(
    candidates: torch.Tensor,
    Z_archive: torch.Tensor,
    R_archive: torch.Tensor,
    n_neighbors: int = 10,
    power: float = 2.0,
) -> torch.Tensor:
    """Score using Inverse Distance Weighting (IDW)."""
    if Z_archive is None or Z_archive.size(0) == 0:
        return score_candidates_random(candidates)

    dists = torch.cdist(candidates, Z_archive)
    k = min(Z_archive.size(0), n_neighbors)
    neigh_dists, neigh_idxs = torch.topk(dists, k=k, largest=False, dim=1)

    weights = 1.0 / (neigh_dists.pow(power) + 1e-8)
    weights = weights / weights.sum(dim=1, keepdim=True)

    neigh_rewards = R_archive[neigh_idxs].squeeze(-1)
    scores = (weights * neigh_rewards).sum(dim=1)
    return scores


def score_candidates_rank_idw(
    candidates: torch.Tensor,
    Z_archive: torch.Tensor,
    R_archive: torch.Tensor,
    n_neighbors: int = 10,
    power: float = 2.0,
) -> torch.Tensor:
    """Score using rank-transformed IDW (more robust to reward scale)."""
    if Z_archive is None or Z_archive.size(0) == 0:
        return score_candidates_random(candidates)

    N = Z_archive.size(0)
    device = candidates.device

    sorted_indices = torch.argsort(R_archive.view(-1))
    ranks = torch.zeros_like(R_archive.view(-1), dtype=torch.float32)
    ranks[sorted_indices] = torch.arange(N, device=device, dtype=torch.float32)
    ranks = ranks / max(1, N - 1)

    dists = torch.cdist(candidates, Z_archive)
    k = min(N, n_neighbors)
    neigh_dists, neigh_idxs = torch.topk(dists, k=k, largest=False, dim=1)

    weights = 1.0 / (neigh_dists.pow(power) + 1e-8)
    weights = weights / weights.sum(dim=1, keepdim=True)

    neigh_ranks = ranks[neigh_idxs]
    scores = (weights * neigh_ranks).sum(dim=1)
    return scores


def score_candidates_local_gradient(
    candidates: torch.Tensor,
    z_center: torch.Tensor,
    Z_archive: torch.Tensor,
    R_archive: torch.Tensor,
    n_neighbors: int = 10,
) -> torch.Tensor:
    """Score using a local gradient estimate from archive neighbours."""
    if Z_archive is None or Z_archive.size(0) < 3:
        return score_candidates_random(candidates)

    d = z_center.shape[0]
    N = Z_archive.size(0)
    k = min(N, max(n_neighbors, d + 1))

    dists_to_center = torch.cdist(z_center.unsqueeze(0), Z_archive).view(-1)
    _, neigh_idxs = torch.topk(dists_to_center, k=k, largest=False)

    Z_neigh = Z_archive[neigh_idxs]
    R_neigh = R_archive.view(-1)[neigh_idxs]

    center_reward = R_neigh.mean()
    reward_diff = R_neigh - center_reward

    directions = Z_neigh - z_center.unsqueeze(0)
    dir_norms = directions.norm(dim=1, keepdim=True).clamp_min(1e-8)
    directions = directions / dir_norms

    gradient = (reward_diff.unsqueeze(1) * directions).sum(dim=0)
    grad_norm = gradient.norm()
    if grad_norm < 1e-8:
        return score_candidates_random(candidates)

    gradient = gradient / grad_norm
    cand_directions = candidates - z_center.unsqueeze(0)
    scores = cand_directions @ gradient
    return scores


def score_candidates_centroid_pull(
    candidates: torch.Tensor,
    Z_archive: torch.Tensor,
    R_archive: torch.Tensor,
    top_k: int = 10,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Score by proximity to a reward-weighted centroid of top points."""
    if Z_archive is None or Z_archive.size(0) == 0:
        return score_candidates_random(candidates)

    N = Z_archive.size(0)
    k = min(N, top_k)

    top_vals, top_idxs = torch.topk(R_archive.view(-1), k=k)
    Z_top = Z_archive[top_idxs]

    weights = torch.softmax(top_vals / temperature, dim=0)
    centroid = (weights.unsqueeze(1) * Z_top).sum(dim=0)

    dists = torch.cdist(candidates, centroid.unsqueeze(0)).view(-1)
    scores = -dists
    return scores


def score_candidates_elite_expansion(
    candidates: torch.Tensor,
    z_center: torch.Tensor,
    Z_archive: torch.Tensor,
    R_archive: torch.Tensor,
    top_k: int = 5,
    local_scope: bool = True,
) -> torch.Tensor:
    """Score by expanding directions from center to elite points."""
    if Z_archive is None or Z_archive.size(0) == 0:
        return score_candidates_random(candidates)

    N = Z_archive.size(0)

    if local_scope and N > top_k * 4:
        n_neighbors = min(N, max(50, top_k * 10))
        dists_to_center = torch.cdist(z_center.unsqueeze(0), Z_archive).view(-1)
        _, neigh_idxs = torch.topk(dists_to_center, k=n_neighbors, largest=False)

        local_R = R_archive.view(-1)[neigh_idxs]
        _, local_top_k_idxs = torch.topk(local_R, k=min(top_k, n_neighbors))

        top_idxs = neigh_idxs[local_top_k_idxs]
    else:
        k = min(N, top_k)
        _, top_idxs = torch.topk(R_archive.view(-1), k=k)

    Z_top = Z_archive[top_idxs]

    directions = Z_top - z_center.unsqueeze(0)
    dir_norms = directions.norm(dim=1, keepdim=True).clamp_min(1e-8)
    directions = directions / dir_norms

    dists = torch.cdist(candidates, Z_top)
    nearest_elite_idx = torch.argmin(dists, dim=1)

    cand_dirs = directions[nearest_elite_idx]
    vec_from_center = candidates - z_center.unsqueeze(0)
    scores = (vec_from_center * cand_dirs).sum(dim=1)
    return scores


def score_candidates_exploration_bonus(
    candidates: torch.Tensor,
    Z_archive: torch.Tensor,
    R_archive: torch.Tensor,
    exploitation_weight: float = 0.5,
    n_neighbors: int = 5,
) -> torch.Tensor:
    """Score with an explicit exploration–exploitation trade-off."""
    if Z_archive is None or Z_archive.size(0) == 0:
        return score_candidates_random(candidates)

    exploit_scores = score_candidates_rank_idw(
        candidates, Z_archive, R_archive, n_neighbors=n_neighbors
    )

    dists = torch.cdist(candidates, Z_archive)
    min_dists = dists.min(dim=1).values
    explore_scores = min_dists / (min_dists.max() + 1e-8)

    exploit_min, exploit_max = exploit_scores.min(), exploit_scores.max()
    if exploit_max - exploit_min > 1e-8:
        exploit_scores = (exploit_scores - exploit_min) / (exploit_max - exploit_min)

    scores = exploitation_weight * exploit_scores + (1 - exploitation_weight) * explore_scores
    return scores


def select_with_diversity(
    candidates: torch.Tensor,
    scores: torch.Tensor,
    k: int,
    diversity_weight: float = 0.3,
) -> torch.Tensor:
    """Select top-k candidates with an explicit diversity bonus."""
    device = candidates.device
    n = candidates.shape[0]
    k = min(k, n)

    if diversity_weight <= 0 or k == 1:
        return torch.topk(scores, k=k).indices

    score_min, score_max = scores.min(), scores.max()
    if score_max - score_min > 1e-8:
        norm_scores = (scores - score_min) / (score_max - score_min)
    else:
        norm_scores = torch.ones_like(scores)

    selected_indices: List[int] = []
    remaining_mask = torch.ones(n, dtype=torch.bool, device=device)

    first_idx = torch.argmax(scores).item()
    selected_indices.append(first_idx)
    remaining_mask[first_idx] = False

    for _ in range(k - 1):
        if not remaining_mask.any():
            break

        remaining_indices = torch.nonzero(remaining_mask, as_tuple=False).view(-1)
        remaining_cands = candidates[remaining_indices]
        remaining_scores = norm_scores[remaining_indices]

        selected_cands = candidates[torch.tensor(selected_indices, device=device)]
        dists = torch.cdist(remaining_cands, selected_cands)
        min_dists = dists.min(dim=1).values

        if min_dists.max() > 1e-8:
            norm_diversity = min_dists / min_dists.max()
        else:
            norm_diversity = torch.zeros_like(min_dists)

        combined = (1 - diversity_weight) * remaining_scores + diversity_weight * norm_diversity

        best_remaining_idx = torch.argmax(combined).item()
        actual_idx = remaining_indices[best_remaining_idx].item()
        selected_indices.append(actual_idx)
        remaining_mask[actual_idx] = False

    return torch.tensor(selected_indices, device=device)


# ==============================================================================
# REGION IMPROVEMENT / ALLOCATION
# ==============================================================================


def compute_region_improvement_rates(
    region_rewards_history: List[List[float]],
    window: int = 3,
) -> List[float]:
    """Compute improvement rates per region for adaptive allocation."""
    rates: List[float] = []
    for history in region_rewards_history:
        if len(history) < 2:
            rates.append(0.0)
            continue

        recent = history[-window:] if len(history) >= window else history
        if len(recent) < 2:
            rates.append(0.0)
            continue

        x = torch.arange(len(recent), dtype=torch.float32)
        y = torch.tensor(recent, dtype=torch.float32)

        x_mean = x.mean()
        y_mean = y.mean()

        numerator = ((x - x_mean) * (y - y_mean)).sum()
        denominator = ((x - x_mean) ** 2).sum()

        if denominator > 1e-8:
            slope = numerator / denominator
            rates.append(float(slope))
        else:
            rates.append(0.0)

    return rates


def adaptive_region_allocation(
    improvement_rates: List[float],
    total_budget: int,
    min_per_region: int = 1,
    temperature: float = 1.0,
) -> List[int]:
    """Allocate budget across regions based on improvement rates."""
    n_regions = len(improvement_rates)

    if total_budget < n_regions * min_per_region:
        return [max(1, total_budget // n_regions)] * n_regions

    rates = torch.tensor(improvement_rates, dtype=torch.float32)
    rates = rates - rates.min() + 0.1

    remaining = total_budget - n_regions * min_per_region
    if remaining <= 0:
        return [min_per_region] * n_regions

    weights = torch.softmax(rates / temperature, dim=0)
    extra = (weights * remaining).floor().int()

    allocated = [min_per_region + int(e) for e in extra]
    diff = total_budget - sum(allocated)

    if diff > 0:
        best_idx = int(torch.argmax(rates).item())
        allocated[best_idx] += diff

    return allocated


# ==============================================================================
# COSINE-BASED HELPERS
# ==============================================================================


def score_candidates_cosine(
    candidates: torch.Tensor,
    z_center: torch.Tensor,
    Z_archive: torch.Tensor,
    R_archive: torch.Tensor,
    top_k: int = 20,
    use_relative: bool = True,
) -> torch.Tensor:
    """Score candidates by cosine similarity to high‑performing history points."""
    if Z_archive is None or Z_archive.size(0) == 0:
        return score_candidates_random(candidates)

    N = Z_archive.size(0)
    k = min(N, top_k)
    top_vals, top_idxs = torch.topk(R_archive.view(-1), k=k)
    Z_top = Z_archive[top_idxs]

    if use_relative:
        target_vecs = Z_top - z_center.unsqueeze(0)
        cand_vecs = candidates - z_center.unsqueeze(0)
    else:
        target_vecs = Z_top
        cand_vecs = candidates

    target_norms = target_vecs.norm(dim=1, keepdim=True).clamp_min(1e-8)
    target_dirs = target_vecs / target_norms

    cand_norms = cand_vecs.norm(dim=1, keepdim=True).clamp_min(1e-8)
    cand_dirs = cand_vecs / cand_norms

    sims = cand_dirs @ target_dirs.T
    scores = sims.max(dim=1).values
    return scores


def calculate_cosine_adaptation(
    current_length: float,
    z_center: torch.Tensor,
    prev_center: torch.Tensor,
    new_best_cand: torch.Tensor,
    min_length: float,
    max_length: float,
    expansion_factor: float = 1.5,
    shrink_factor: float = 0.5,
    threshold: float = 0.0,
) -> float:
    """Adapt trust-region length based on directional consistency (cosine)."""
    if prev_center is None:
        return current_length

    v_prev = z_center - prev_center
    v_curr = new_best_cand - z_center

    norm_prev = v_prev.norm()
    norm_curr = v_curr.norm()

    if norm_prev < 1e-8 or norm_curr < 1e-8:
        return current_length

    cos_sim = (v_prev @ v_curr) / (norm_prev * norm_curr)

    if cos_sim > threshold:
        return max(min_length, min(max_length, current_length * expansion_factor))
    if cos_sim < -0.5:
        return max(min_length, min(max_length, current_length * shrink_factor))
    return current_length


# ==============================================================================
# SCORING REGISTRY
# ==============================================================================


SCORING_METHODS: Dict[str, Callable] = {
    "random": lambda c, zc, Za, Ra, pc, gen, **kw: score_candidates_random(c, gen),
    "linear": lambda c, zc, Za, Ra, pc, gen, **kw: score_candidates_linear(
        c, zc, Za, Ra, n_neighbors=int(kw.get("n_neighbors", 20)), regularization=float(kw.get("regularization", 1e-4))
    ),
    "momentum": lambda c, zc, Za, Ra, pc, gen, **kw: score_candidates_momentum(c, zc, pc if pc is not None else zc),
    "idw": lambda c, zc, Za, Ra, pc, gen, **kw: score_candidates_idw(
        c, Za, Ra, n_neighbors=int(kw.get("n_neighbors", 10)), power=float(kw.get("idw_power", 2.0))
    ),
    "rank_idw": lambda c, zc, Za, Ra, pc, gen, **kw: score_candidates_rank_idw(
        c, Za, Ra, n_neighbors=int(kw.get("n_neighbors", 10)), power=float(kw.get("idw_power", 2.0))
    ),
    "local_gradient": lambda c, zc, Za, Ra, pc, gen, **kw: score_candidates_local_gradient(
        c, zc, Za, Ra, n_neighbors=int(kw.get("n_neighbors", 10))
    ),
    "centroid_pull": lambda c, zc, Za, Ra, pc, gen, **kw: score_candidates_centroid_pull(
        c, Za, Ra, top_k=int(kw.get("centroid_top_k", 10)), temperature=float(kw.get("centroid_temperature", 1.0))
    ),
    "elite_expansion": lambda c, zc, Za, Ra, pc, gen, **kw: score_candidates_elite_expansion(
        c, zc, Za, Ra, top_k=int(kw.get("top_k", 5)), local_scope=bool(kw.get("local_scope", True))
    ),
    "exploration_bonus": lambda c, zc, Za, Ra, pc, gen, **kw: score_candidates_exploration_bonus(
        c, Za, Ra, exploitation_weight=float(kw.get("exploitation_weight", 0.5)), n_neighbors=int(kw.get("n_neighbors", 5))
    ),
    "cosine": lambda c, zc, Za, Ra, pc, gen, **kw: score_candidates_cosine(
        c, zc, Za, Ra, top_k=int(kw.get("top_k", 20)), use_relative=bool(kw.get("use_relative", True))
    ),
}


def get_scoring_function(method_name: str) -> Callable:
    """Return a scoring function by name (defaults to random)."""
    return SCORING_METHODS.get(method_name.lower(), SCORING_METHODS["random"])


# ==============================================================================
# CANDIDATE GENERATION
# ==============================================================================


def generate_candidates_fast_forward(
    z_center: torch.Tensor,
    length: float,
    n_candidates: int,
    device: str,
    prob_perturb: Optional[float] = None,
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    tr_shape: str = "hypercube",
    pool_size: int = 1024,
) -> torch.Tensor:
    """Generate candidates using Sobol fast_forward optimization for random selection.
    
    Instead of generating a full Sobol pool and scoring, we use fast_forward to jump
    to a random position in the Sobol sequence. This is much faster when using random scoring.
    
    To match the behavior of picking n_candidates randomly from a pool of pool_size,
    we sample a random offset in [0, pool_size - n_candidates] and draw n_candidates
    sequential samples from there.
    
    Args:
        z_center: Center point in reduced space, shape (d,)
        length: Trust region side length
        n_candidates: Number of candidate points to generate
        device: Device for tensor operations
        prob_perturb: Probability of perturbing each dimension (None to disable masking)
        seed: Random seed for Sobol sequence generation
        generator: Optional PyTorch generator for reproducible randomness
        tr_shape: Trust region shape (only hypercube supported for fast_forward)
        pool_size: Maximum Sobol index (pool size) to stay within
        
    Returns:
        Candidate points, shape (n_candidates, d)
    """
    d = int(z_center.numel())

    
    # Safeguard for high dimension
    if d > 21201:
        #print(f"[TRS] WARNING: Sobol fast_forward does not support dim={d} > 21201. Falling back to standard Gaussian candidates.")
        return generate_candidates_around_center(
            z_center=z_center, length=length, n_candidates=n_candidates,
            device=device, prob_perturb=prob_perturb, seed=seed,
            sampling_mode="gaussian_box", generator=generator, tr_shape=tr_shape
        )

    sobol_seed = seed if seed is not None else 42
    
    # Hypercube bounds
    lb = z_center - (length / 2.0)
    ub = z_center + (length / 2.0)
    
    # Sample a single random offset to stay within [0, pool_size]
    max_offset = max(0, pool_size - n_candidates)
    if generator is not None:
        offset = int(torch.randint(0, max_offset + 1, (1,), generator=generator, device=device).item())
    else:
        offset = int(torch.randint(0, max_offset + 1, (1,)).item())
    
    # Create SobolEngine and fast_forward to the offset
    sob = get_sobol_engine(d, scramble=True, seed=sobol_seed)
    sob.fast_forward(offset)
    
    # Draw n_candidates sequential samples
    U = sob.draw(n_candidates).to(dtype=torch.float32, device=device)
    
    # Scale to trust region bounds
    C = lb + (ub - lb) * U
    
    # Apply perturbation mask if needed
    if prob_perturb is not None:
        p = float(max(0.0, min(1.0, prob_perturb)))
        if generator is not None:
            mask = (torch.rand(n_candidates, d, device=device, generator=generator) < p).to(C.dtype)
        else:
            mask = (torch.rand(n_candidates, d, device=device) < p).to(C.dtype)
        
        rows_all_zero = (mask.sum(dim=1) == 0)
        if rows_all_zero.any():
            idxs = torch.nonzero(rows_all_zero, as_tuple=False).view(-1)
            if generator is not None:
                rand_cols = torch.randint(
                    low=0, high=d, size=(idxs.numel(),),
                    device=device, generator=generator
                )
            else:
                rand_cols = torch.randint(
                    low=0, high=d, size=(idxs.numel(),), device=device
                )
            mask[idxs, rand_cols] = 1.0
        
        C = z_center.unsqueeze(0) + (C - z_center.unsqueeze(0)) * mask
    
    return C


def generate_candidates_around_center(
    z_center: torch.Tensor,
    length: float,
    n_candidates: int,
    device: str,
    prob_perturb: Optional[float] = None,
    seed: Optional[int] = None,
    sampling_mode: str = "sobol",
    generator: Optional[torch.Generator] = None,
    tr_shape: str = "hypercube",
) -> torch.Tensor:
    """Generate candidates around a trust region center using Sobol or Gaussian sampling.
    
    Args:
        z_center: Center point in reduced space, shape (d,)
        length: Trust region side length (for hypercube/hyperrectangle) or diameter (for hypersphere)
        n_candidates: Number of candidate points to generate
        device: Device for tensor operations
        prob_perturb: Probability of perturbing each dimension (None to disable masking)
        seed: Random seed for Sobol sequence generation
        sampling_mode: "sobol" or "gaussian" for candidate generation (ignored for hypersphere)
        generator: Optional PyTorch generator for reproducible randomness
        tr_shape: "hypercube" (same length in all dimensions), "hyperrectangle" (scaled per dimension),
                  or "hypersphere" (uniform sampling from sphere/disk)
        
    Returns:
        Candidate points, shape (n_candidates, d)
    """
    d = int(z_center.numel())

    
    # Safeguard: Fallback to Gaussian if Sobol requested but dimension is too high
    if sampling_mode.lower() == "sobol" and d > 21201:
        #print(f"[TRS] WARNING: SobolEngine does not support dim={d} > 21201. Falling back to Gaussian.")
        sampling_mode = "gaussian"

    if tr_shape.lower() == "hyperrectangle":
        # Hyperrectangle: Different side lengths per dimension (axis-aligned box)
        # This creates an elongated shape where different dimensions have different exploration ranges
        # Use a simple scaling: scale by dimension index to create variation
        # Scale factors range from 0.7 to 1.3, creating a ~2:1 aspect ratio variation
        dim_indices = torch.arange(d, device=device, dtype=torch.float32)
        # Normalize to [0, 1] then map to [0.7, 1.3] for scaling factors
        normalized = dim_indices / max(1, d - 1) if d > 1 else torch.tensor([0.5], device=device)
        dim_factors = 0.7 + 0.6 * normalized  # Range: [0.7, 1.3]
        lengths_per_dim = length * dim_factors
        lb = z_center - (lengths_per_dim / 2.0)
        ub = z_center + (lengths_per_dim / 2.0)
    elif tr_shape.lower() == "hypersphere":
        # Hypersphere: Uniform sampling from a sphere/disk
        # This is handled separately in the sampling section below
        # We'll use the length parameter as the radius
        pass
    else:
        # Hypercube: Same length in all dimensions (default, original behavior)
        # Sobol samples in [0,1]^d are scaled uniformly: creates an axis-aligned cube
        lb = z_center - (length / 2.0)
        ub = z_center + (length / 2.0)

    # Generate candidates based on sampling mode and trust region shape
    if tr_shape.lower() == "hypersphere":
        # Hypersphere: Uniform sampling from a sphere/disk
        # Method: Sample direction uniformly on unit sphere, then scale by radius
        # For uniform volume sampling: r ~ U[0,1]^(1/d) where d is dimension
        radius = length / 2.0  # Use length/2 as radius (so length is diameter)
        
        if generator is not None:
            # Sample direction: sample from standard Gaussian, normalize to unit sphere
            direction = torch.randn(n_candidates, d, device=device, generator=generator, dtype=torch.float32)
            # Normalize to unit sphere
            direction_norm = direction.norm(dim=1, keepdim=True)
            direction = direction / (direction_norm + 1e-12)  # Avoid division by zero
            
            # Sample radius uniformly in volume: r ~ U[0,1]^(1/d)
            # For uniform volume, P(r < R) = R^d, so r = U^(1/d) where U ~ Uniform[0,1]
            r_uniform = torch.rand(n_candidates, 1, device=device, generator=generator, dtype=torch.float32)
            r = torch.pow(r_uniform, 1.0 / d) * radius
            
            C = z_center.unsqueeze(0) + r * direction
        else:
            # Same without generator
            direction = torch.randn(n_candidates, d, device=device, dtype=torch.float32)
            direction_norm = direction.norm(dim=1, keepdim=True)
            direction = direction / (direction_norm + 1e-12)
            
            r_uniform = torch.rand(n_candidates, 1, device=device, dtype=torch.float32)
            r = torch.pow(r_uniform, 1.0 / d) * radius
            
            C = z_center.unsqueeze(0) + r * direction
    elif sampling_mode.lower() in ("gaussian", "gaussian_box"):
        # Gaussian sampling: sample from N(z_center, sigma^2)
        # Modes:
        # - "gaussian": uses length/2 as sigma (standard implementation)
        # - "gaussian_box": uses length/sqrt(12) to match per-coordinate variance 
        #   of uniform sampling in a box of side length 'length'
        if sampling_mode.lower() == "gaussian_box":
            sigma = length / np.sqrt(12.0)
        else:
            sigma = length / 2.0

        if generator is not None:
            C = z_center.unsqueeze(0) + sigma * torch.randn(
                n_candidates, d, device=device, 
                generator=generator, dtype=torch.float32
            )
        else:
            C = z_center.unsqueeze(0) + sigma * torch.randn(
                n_candidates, d, device=device, dtype=torch.float32
            )

        # mask with probability prob_perturb
        if prob_perturb is not None:
            p = float(max(0.0, min(1.0, prob_perturb)))
            if generator is not None:
                mask = (torch.rand(n_candidates, d, device=device, generator=generator) < p).to(C.dtype)
            else:
                mask = (torch.rand(n_candidates, d, device=device) < p).to(C.dtype)
            
            # Apply mask: C = center + (C - center) * mask
            C = z_center.unsqueeze(0) + (C - z_center.unsqueeze(0)) * mask
        
        # Clip to trust region bounds [z_center - length/2, z_center + length/2]
        # Only clip if not hypersphere (hypersphere doesn't use lb/ub)
        if tr_shape.lower() != "hypersphere":
            C = torch.clamp(C, min=lb.unsqueeze(0), max=ub.unsqueeze(0))
    else:
        # Sobol sampling (default): generates low-discrepancy sequences
        # Note: Sobol naturally samples from hypercube, so this works for hypercube/hyperrectangle
        # but not for hypersphere (which uses the branch above)
        sobol_seed = seed if seed is not None else 42

        # Generate Sobol samples in [0,1]^d and scale to [lb, ub]
        # Use cached engine to avoid heavy initialization in high dimensions
        sob = get_sobol_engine(d, scramble=True, seed=sobol_seed)
        
        # Note: If not using fast_forward, draw() advances the engine state.
        # This is generally fine for standard TRS logic.
        U = sob.draw(n_candidates).to(dtype=torch.float32, device=device)
        C = lb + (ub - lb) * U

    # Apply perturbation mask: hold some dimensions fixed at center
    # This allows exploring only a subset of dimensions per candidate
    # Note: If sampling_mode was Gaussian, we already applied the mask above.
    # We only apply it here if it wasn't already applied.
    if prob_perturb is not None and sampling_mode.lower() not in ("gaussian", "gaussian_box"):
        p = float(max(0.0, min(1.0, prob_perturb)))
    
        # Create binary mask: 1 = perturb, 0 = keep at center
        if generator is not None:
            mask = (torch.rand(n_candidates, d, device=device, generator=generator) < p).to(C.dtype)
        else:
            mask = (torch.rand(n_candidates, d, device=device) < p).to(C.dtype)
        
        # Ensure at least one dimension is perturbed per candidate
        # (prevents degenerate candidates identical to center)
        rows_all_zero = (mask.sum(dim=1) == 0)
        if rows_all_zero.any():
            idxs = torch.nonzero(rows_all_zero, as_tuple=False).view(-1)
            if generator is not None:
                rand_cols = torch.randint(
                    low=0, high=d, size=(idxs.numel(),), 
                    device=device, generator=generator
                )
            else:
                rand_cols = torch.randint(
                    low=0, high=d, size=(idxs.numel(),), device=device
                )
            mask[idxs, rand_cols] = 1.0
        
        # Apply mask: C = center + (C - center) * mask
        C = z_center.unsqueeze(0) + (C - z_center.unsqueeze(0)) * mask
    
    return C


# ==============================================================================
# WARMUP / INITIALIZATION HELPERS
# ==============================================================================


def generate_warmup_samples(
    n_samples: int,
    dim: int,
    sampling_mode: str,
    device: str,
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate warmup samples using Sobol or Gaussian sampling.
    
    Args:
        n_samples: Number of samples to generate
        dim: Dimensionality of the space
        sampling_mode: "sobol" or "gaussian"
        device: Device for tensor operations
        seed: Random seed for Sobol sequence
        generator: PyTorch generator for Gaussian sampling
        
    Returns:
        Samples of shape (n_samples, dim)
    """
    if sampling_mode.lower() == "sobol" and dim > 21201:
        #print(f"[TRS] WARNING: SobolEngine does not support dim={dim} > 21201. Falling back to Gaussian warmup.")
        sampling_mode = "gaussian"

    if sampling_mode.lower() == "sobol":
        # Sobol sampling: generate uniform samples in [0,1] then transform to Gaussian
        # Use cached engine to avoid heavy initialization
        sob = get_sobol_engine(dim, scramble=True, seed=seed if seed is not None else 42)
        U = sob.draw(n_samples).to(dtype=torch.float32, device=device)
        # Transform uniform [0,1] to standard Gaussian via inverse CDF
        z_init = torch.sqrt(torch.tensor(2.0, device=device, dtype=torch.float32)) * torch.special.erfinv(2.0 * U - 1.0)
    else:
        # Gaussian initialization: sample directly from standard normal
        if generator is not None:
            z_init = torch.randn(n_samples, dim, device=device, generator=generator)
        else:
            z_init = torch.randn(n_samples, dim, device=device)
    return z_init


# ==============================================================================
# CANDIDATE GENERATION AND SELECTION
# ==============================================================================


def generate_and_select_candidates(
    z_center: torch.Tensor,
    tr_state: Any,  # _TRState object
    generate_candidates_fn: Callable,
    Z_archive: torch.Tensor,
    R_archive: torch.Tensor,
    prev_center: torch.Tensor,
    k_evals: int,
    pool_per_region: int,
    prob_perturb_kw: Optional[float],
    perturb_min_frac: float,
    perturb_max_frac: float,
    search_sampling_mode: str,
    region_seed: Optional[int],
    generator: Optional[torch.Generator],
    scoring_method: str,
    heuristic_kwargs: Dict[str, Any],
    use_two_phase: bool,
    phase_switch_frac: float,
    oracle_budget: int,
    num_iterations: int,
    batch_size: int,
    n_eval: int,
    use_diversity_selection: bool,
    diversity_weight: float,
    use_fast_forward: bool = False,
    max_sobol_index: int = 1024,
    pregenerated_U: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generate candidate pool and select top-k candidates for a trust region.
    
    Args:
        z_center: Trust region center in z-space
        tr_state: Trust region state object
        generate_candidates_fn: Function to generate candidates around center
        Z_archive: Archive of z-space points
        R_archive: Archive of rewards
        prev_center: Previous center for momentum calculations
        k_evals: Number of candidates to select
        pool_per_region: Size of candidate pool
        prob_perturb_kw: Fixed perturbation probability (None for adaptive)
        perturb_min_frac: Minimum perturbation fraction
        perturb_max_frac: Maximum perturbation fraction
        search_sampling_mode: "sobol" or "gaussian"
        region_seed: Seed for this region
        generator: PyTorch generator
        scoring_method: Scoring method name
        heuristic_kwargs: Additional kwargs for scoring function
        use_two_phase: Whether to use two-phase scoring
        phase_switch_frac: Fraction of budget before switching phases
        oracle_budget: Total oracle budget
        num_iterations: Number of iterations
        batch_size: Batch size per iteration
        n_eval: Current number of evaluations
        use_diversity_selection: Whether to use diversity selection
        diversity_weight: Weight for diversity in selection
        use_fast_forward: Whether to use fast-forward optimization
        max_sobol_index: Maximum Sobol index for fast-forward
        pregenerated_U: Optional pre-generated Sobol points in [0, 1]^d
        
    Returns:
        Selected candidates of shape (k_evals, dim)
    """
    # Determine scoring method (two-phase logic) - needed for fast_forward check
    if use_two_phase:
        total_budget = oracle_budget if oracle_budget > 0 else (num_iterations * batch_size)
        phase_switch_point = int(total_budget * phase_switch_frac)
        current_scoring = "random" if n_eval < phase_switch_point else scoring_method
    else:
        current_scoring = scoring_method
    
    # Determine perturbation probability
    if prob_perturb_kw is None:
        if generator is not None:
            p = float(perturb_min_frac + (perturb_max_frac - perturb_min_frac) * torch.rand(1, device=z_center.device, generator=generator).item())
        else:
            p = float(perturb_min_frac + (perturb_max_frac - perturb_min_frac) * torch.rand(1, device=z_center.device).item())
    else:
        try:
            p = float(prob_perturb_kw)
        except Exception:
            p = tr_state.length / max(1e-12, tr_state.max_length)
        p = float(max(perturb_min_frac, min(perturb_max_frac, p)))
    
    # Use pre-generated Sobol points if available
    if pregenerated_U is not None:
        d = int(z_center.numel())
        device = z_center.device
        
        # Scale to bounds
        lb = z_center - (tr_state.length / 2.0)
        ub = z_center + (tr_state.length / 2.0)
        
        # Scale U to bounds
        Z_pool = lb + (ub - lb) * pregenerated_U
        
        # Apply perturbation mask
        if p < 1.0:
            if generator is not None:
                mask = (torch.rand(Z_pool.shape[0], d, device=device, generator=generator) < p).to(Z_pool.dtype)
            else:
                mask = (torch.rand(Z_pool.shape[0], d, device=device) < p).to(Z_pool.dtype)
            
            rows_all_zero = (mask.sum(dim=1) == 0)
            if rows_all_zero.any():
                idxs = torch.nonzero(rows_all_zero, as_tuple=False).view(-1)
                if generator is not None:
                    rand_cols = torch.randint(low=0, high=d, size=(idxs.numel(),), device=device, generator=generator)
                else:
                    rand_cols = torch.randint(low=0, high=d, size=(idxs.numel(),), device=device)
                mask[idxs, rand_cols] = 1.0
            
            Z_pool = z_center.unsqueeze(0) + (Z_pool - z_center.unsqueeze(0)) * mask
            
        # If random selection path (pregenerated_U matches k_evals)
        if current_scoring == "random" and not use_diversity_selection and Z_pool.shape[0] == k_evals:
            return Z_pool
        
        # Otherwise, fall through to scoring and selection using Z_pool
    
    # Standard path if no pre-generated points or scoring is required
    elif use_fast_forward and current_scoring == "random" and search_sampling_mode.lower() == "sobol":
        # Use fast_forward to generate exactly k_evals candidates (no need for full pool)
        # We sample a random starting point within the first 'pool_per_region' samples
        sobol_seed = region_seed if region_seed is not None else 42
        
        # Generate candidates directly using fast_forward
        # We use pool_per_region as the effective size of the virtual pool
        chosen = generate_candidates_fast_forward(
            z_center=z_center,
            length=tr_state.length,
            n_candidates=k_evals,
            device=z_center.device,
            prob_perturb=p,
            seed=sobol_seed,
            generator=generator,
            tr_shape="hypercube",  # fast_forward only supports hypercube
            pool_size=pool_per_region,
        )
        return chosen
    
    if pregenerated_U is None:
        # Standard pool generation if not already handled by fast_forward or pregenerated_U
        Z_pool = generate_candidates_fn(
            z_center=z_center,
            length=tr_state.length,
            n_candidates=pool_per_region,
            prob_perturb=p,
            seed=region_seed,
            sampling_mode=search_sampling_mode,
        )
    
    # Score candidates
    scoring_func = get_scoring_function(current_scoring)
    scores = scoring_func(
        Z_pool, z_center, Z_archive, R_archive, prev_center, generator, **heuristic_kwargs
    )
    
    # Select top candidates
    if use_diversity_selection and k_evals > 1:
        selected_indices = select_with_diversity(
            Z_pool, scores, k=k_evals, diversity_weight=diversity_weight
        )
        chosen = Z_pool[selected_indices]
    else:
        topk = torch.topk(scores, k=min(k_evals, scores.shape[0]))
        chosen = Z_pool[topk.indices]
    
    return chosen


# ==============================================================================
# TRUST REGION STATE UPDATES
# ==============================================================================


def update_trust_region_states(
    regions_data: List[Tuple[int, torch.Tensor, int]],
    all_y_true: torch.Tensor,
    all_x0: torch.Tensor,
    states: List[Any],  # List of _TRState objects
    centers_z: torch.Tensor,
    Z_archive: torch.Tensor,
    R_archive: torch.Tensor,
    init_length: float,
    use_adaptive_allocation: bool,
    region_rewards_history: Optional[List[List[float]]] = None,
    return_latents: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], Optional[List[torch.Tensor]]]:
    """Update trust region states after batch evaluation.
    
    Args:
        regions_data: List of (region_idx, chosen_candidates, k_evals) tuples
        all_y_true: All evaluation results, shape (total_candidates,)
        all_x0: All evaluated points in latent space
        states: List of trust region state objects
        centers_z: Current trust region centers
        Z_archive: Archive of z-space points
        R_archive: Archive of rewards
        init_length: Initial trust region length
        use_adaptive_allocation: Whether to track region rewards
        region_rewards_history: Optional list to append region rewards
        return_latents: Whether to return the full latent tensors (x0) for each region
        
    Returns:
        Tuple of (new_z_all, new_y_all, Optional[new_x_all]) lists
    """
    new_z_all = []
    new_y_all = []
    new_x_all = [] if return_latents else None
    offset = 0
    
    for ridx, chosen, k_evals in regions_data:
        # Extract this region's evaluation results
        y_true = all_y_true[offset:offset + k_evals]
        offset += k_evals
        
        # Skip update if no candidates were evaluated for this region
        if y_true.numel() == 0:
            new_z_all.append(torch.empty((0, chosen.shape[1]), device=chosen.device, dtype=chosen.dtype))
            new_y_all.append(y_true.detach().clone())
            if return_latents and new_x_all is not None:
                new_x_all.append(torch.empty((0, *all_x0.shape[1:]), device=all_x0.device, dtype=all_x0.dtype))
            continue

        st = states[ridx]
        
        # Update trust region state based on best in this batch
        best_loc_val, best_loc_idx = torch.max(y_true, dim=0)
        st.update(float(best_loc_val.item()))
        
        # Track region rewards for adaptive allocation
        if use_adaptive_allocation and region_rewards_history is not None:
            region_rewards_history[ridx].append(float(best_loc_val.item()))
        
        # Handle restart trigger (reset to global best)
        if st.restart_triggered:
            st.length = init_length
            st.restart_triggered = False
            gb_idx = int(torch.argmax(R_archive).item())
            centers_z[ridx] = Z_archive[gb_idx].detach().clone()
        
        new_z_all.append(chosen.detach().clone())
        new_y_all.append(y_true.detach().clone())
        if return_latents and new_x_all is not None:
            x0 = all_x0[offset - k_evals:offset]
            new_x_all.append(x0.detach().clone())
    
    return new_z_all, new_y_all, new_x_all


# ==============================================================================
# CENTER SELECTION WRAPPER
# ==============================================================================


def update_trust_region_centers(
    Z_archive: torch.Tensor,
    R_archive: torch.Tensor,
    num_regions: int,
    center_selection_mode: str,
    tr_config: Dict[str, Any],
    device: str,
    Z_last: Optional[torch.Tensor] = None,
    R_last: Optional[torch.Tensor] = None,
    new_z_all: Optional[List[torch.Tensor]] = None,
    new_y_all: Optional[List[torch.Tensor]] = None,
    current_centers_z: Optional[torch.Tensor] = None,
    current_centers_val: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Update trust region centers based on archive or last iteration results.
    
    Args:
        Z_archive: Archive of z-space points
        R_archive: Archive of rewards
        num_regions: Number of trust regions
        center_selection_mode: "topk"/"global_topk", "diverse", "clustering", "last_iter_topk", 
                               "strict_local", or "last_iter_local" (case-insensitive)
        tr_config: Trust region configuration dict
        device: Device for tensor operations
        Z_last: Optional batch of points from the last iteration
        R_last: Optional rewards from the last iteration
        new_z_all: Optional list of points per region from last iteration
        new_y_all: Optional list of rewards per region from last iteration
        current_centers_z: Current centers in z-space
        current_centers_val: Current center values
        
    Returns:
        Tuple of (centers_z, centers_val)
    """
    kcent = min(num_regions, int(R_archive.shape[0]))
    
    # Normalize mode to lowercase for case-insensitive matching
    mode = str(center_selection_mode).lower()
    if mode in ("topk", "global_topk", "global", "global_best"):
        mode = "topk"
    
    if mode == 'diverse':
        new_centers, new_vals = select_centers_diverse(
            Z_archive, R_archive, kcent,
            min_dist=float(tr_config.get('min_center_dist', 0.1)),
            device=device
        )
    elif mode == 'clustering':
        new_centers, new_vals = select_centers_clustering(
            Z_archive, R_archive, kcent,
            top_percentile=float(tr_config.get('clustering_percentile', 0.2)),
            device=device
        )
    elif mode == 'last_iter_topk' and R_last is not None and Z_last is not None:
        # Strategy 4: k best points from last iteration
        kcent_last = min(num_regions, int(R_last.shape[0]))
        l_top = torch.topk(R_last.view(-1), k=kcent_last)
        new_centers = Z_last[l_top.indices].detach().clone()
        new_vals = l_top.values.detach().clone()
        
        # If we have fewer than kcent points in last iteration, fill with global top-k
        if kcent_last < num_regions:
            g_top = torch.topk(R_archive.view(-1), k=num_regions)
            # Find points not already in new_centers (simplified: just take top ones)
            # For simplicity, we just append or replace
            fill_k = num_regions - kcent_last
            new_centers = torch.cat([new_centers, Z_archive[g_top.indices[:fill_k]]], dim=0)
            new_vals = torch.cat([new_vals, g_top.values[:fill_k]], dim=0)

    elif mode == 'last_iter_local' and new_z_all is not None and new_y_all is not None:
        # Strategy 2: best point from last iteration for each region
        new_centers_list = []
        new_vals_list = []
        for i in range(min(num_regions, len(new_z_all))):
            if new_y_all[i].numel() > 0:
                best_idx = torch.argmax(new_y_all[i])
                new_centers_list.append(new_z_all[i][best_idx].unsqueeze(0))
                new_vals_list.append(new_y_all[i][best_idx].unsqueeze(0))
            elif current_centers_z is not None and i < current_centers_z.shape[0]:
                # Fallback to current center if no new points for this region
                new_centers_list.append(current_centers_z[i].unsqueeze(0))
                new_vals_list.append(current_centers_val[i].unsqueeze(0))
        
        if len(new_centers_list) > 0:
            new_centers = torch.cat(new_centers_list, dim=0)
            new_vals = torch.cat(new_vals_list, dim=0)
        else:
            # Fallback to top-k
            g_top = torch.topk(R_archive.view(-1), k=kcent)
            new_centers = Z_archive[g_top.indices].detach().clone()
            new_vals = g_top.values.detach().clone()

    elif mode == 'strict_local' and new_z_all is not None and new_y_all is not None:
        # Strategy 1: best observed point ever for each region
        # Note: current_centers_z/val are expected to hold the "best ever" so far
        new_centers_list = []
        new_vals_list = []
        for i in range(min(num_regions, len(new_z_all))):
            if new_y_all[i].numel() > 0:
                best_loc_val, best_loc_idx = torch.max(new_y_all[i], dim=0)
                if current_centers_val is not None and i < current_centers_val.shape[0]:
                    if best_loc_val > current_centers_val[i]:
                        new_centers_list.append(new_z_all[i][best_loc_idx].unsqueeze(0))
                        new_vals_list.append(best_loc_val.unsqueeze(0))
                    else:
                        new_centers_list.append(current_centers_z[i].unsqueeze(0))
                        new_vals_list.append(current_centers_val[i].unsqueeze(0))
                else:
                    new_centers_list.append(new_z_all[i][best_loc_idx].unsqueeze(0))
                    new_vals_list.append(best_loc_val.unsqueeze(0))
            elif current_centers_z is not None and i < current_centers_z.shape[0]:
                new_centers_list.append(current_centers_z[i].unsqueeze(0))
                new_vals_list.append(current_centers_val[i].unsqueeze(0))

        if len(new_centers_list) > 0:
            new_centers = torch.cat(new_centers_list, dim=0)
            new_vals = torch.cat(new_vals_list, dim=0)
        else:
            # Fallback
            g_top = torch.topk(R_archive.view(-1), k=kcent)
            new_centers = Z_archive[g_top.indices].detach().clone()
            new_vals = g_top.values.detach().clone()

    else:
        # Default global top-k (Strategy 3)
        g_top = torch.topk(R_archive.view(-1), k=kcent)
        new_centers = Z_archive[g_top.indices].detach().clone()
        new_vals = g_top.values.detach().clone()
    
    return new_centers, new_vals


# ==============================================================================
# BATCH ALLOCATION
# ==============================================================================


def allocate_batch_across_regions(
    batch_size: int,
    num_regions: int,
    use_adaptive_allocation: bool,
    improvement_rates: Optional[List[float]] = None,
    allocation_temperature: float = 1.0,
) -> List[int]:
    """Allocate batch size across trust regions.
    
    Args:
        batch_size: Total batch size to allocate
        num_regions: Number of trust regions
        use_adaptive_allocation: Whether to use adaptive allocation
        improvement_rates: Improvement rates per region (for adaptive allocation)
        allocation_temperature: Temperature for softmax in adaptive allocation
        
    Returns:
        List of allocations per region
    """
    if use_adaptive_allocation and improvement_rates is not None:
        return adaptive_region_allocation(
            improvement_rates, batch_size,
            min_per_region=1, temperature=allocation_temperature
        )
    else:
        base_per_region = max(1, batch_size // num_regions)
        remainder = max(0, batch_size - base_per_region * num_regions)
        alloc: List[int] = []
        for i in range(num_regions):
            k = base_per_region + (1 if i < remainder else 0)
            alloc.append(k)
        return alloc


