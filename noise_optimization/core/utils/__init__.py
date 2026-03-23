from .seed import set_global_seed, make_generator
from .tr_utils import (  # re-export for convenience
    adaptive_region_allocation,
    allocate_batch_across_regions,
    calculate_1_5_rule_length,
    calculate_cosine_adaptation,
    calculate_variance_based_length,
    compute_region_improvement_rates,
    generate_and_select_candidates,
    generate_candidates_around_center,
    generate_candidates_fast_forward,
    generate_warmup_samples,
    get_scoring_function,
    score_candidates_centroid_pull,
    score_candidates_cosine,
    score_candidates_elite_expansion,
    score_candidates_exploration_bonus,
    score_candidates_idw,
    score_candidates_linear,
    score_candidates_local_gradient,
    score_candidates_momentum,
    score_candidates_random,
    score_candidates_rank_idw,
    select_centers_clustering,
    select_centers_diverse,
    select_with_diversity,
    update_trust_region_centers,
    update_trust_region_states,
)
from .tr_centers import CenterSelectionStrategy
from .tr_archive import update_archive
from .tr_state import TRState

__all__ = [
    "set_global_seed",
    "make_generator",
    # trust-region utilities
    "adaptive_region_allocation",
    "allocate_batch_across_regions",
    "calculate_1_5_rule_length",
    "calculate_cosine_adaptation",
    "calculate_variance_based_length",
    "compute_region_improvement_rates",
    "generate_and_select_candidates",
    "generate_candidates_around_center",
    "generate_candidates_fast_forward",
    "generate_warmup_samples",
    "get_scoring_function",
    "score_candidates_centroid_pull",
    "score_candidates_cosine",
    "score_candidates_elite_expansion",
    "score_candidates_exploration_bonus",
    "score_candidates_idw",
    "score_candidates_linear",
    "score_candidates_local_gradient",
    "score_candidates_momentum",
    "score_candidates_random",
    "score_candidates_rank_idw",
    "select_centers_clustering",
    "select_centers_diverse",
    "select_with_diversity",
    "update_archive",
    "update_trust_region_centers",
    "update_trust_region_states",
    # center selection strategy
    "CenterSelectionStrategy",
    # trust region state
    "TRState",
]


