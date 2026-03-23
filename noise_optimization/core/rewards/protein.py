from __future__ import annotations

import os
import sys
import tempfile
import time
import hashlib
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from loguru import logger

from .base import RewardConfig, RewardFunction, RewardFunctionRegistry
from ..utils.path_utils import setup_proteina_path


def _setup_proteina_path():
    """Add proteina directory to sys.path for proteinfoundation imports.
    
    Backward compatibility wrapper for the new path_utils utility.
    """
    return setup_proteina_path()


def _resolve_cache_dir(config: RewardConfig, default_name: str, *, unique_run: bool = False) -> Path:
    """Resolve a cache directory for protein rewards.

    IMPORTANT: Some rewards (e.g., PLDDTReward) spawn subprocesses and may run multiple
    experiments/jobs concurrently on the same filesystem. If they share a single cache
    directory, cleanup can delete other jobs' temporary PDBs. Setting unique_run=True
    isolates each run in a unique subdirectory to avoid cross-job interference.
    """
    cache_root = config.cache_dir or os.path.join(os.getcwd(), default_name)
    base_path = Path(cache_root)
    base_path.mkdir(parents=True, exist_ok=True)

    if not unique_run:
        return base_path

    slurm_job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID") or "nojid"
    run_id = f"run_{slurm_job_id}_{os.getpid()}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    run_path = base_path / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path


def _extract_atom37_list(candidates: Any) -> List[torch.Tensor]:
    """Normalize reward inputs to a list of atom37 tensors with shape [N, 37, 3]."""
    tensors: List[torch.Tensor] = []

    def _append_tensor(t: torch.Tensor) -> None:
        t = t.detach().cpu().float()
        if t.dim() == 4:
            for i in range(t.shape[0]):
                tensors.append(t[i])
        elif t.dim() == 3:
            tensors.append(t)
        else:
            raise ValueError(f"Expected atom37 tensor with 3 or 4 dims, got {t.shape}")

    if isinstance(candidates, torch.Tensor):
        _append_tensor(candidates)
    elif isinstance(candidates, list):
        for entry in candidates:
            if isinstance(entry, torch.Tensor):
                _append_tensor(entry)
            else:
                raise TypeError("Protein rewards expect tensors representing atom37 coordinates.")
    else:
        raise TypeError("Protein rewards expect tensors or list of tensors as inputs.")

    return tensors


def _extract_ca_coords_list(candidates: Any) -> List[np.ndarray]:
    """Extract CA (alpha carbon) coordinates from atom37 tensors.
    
    Returns list of numpy arrays with shape [N, 3] containing CA coordinates.
    CA atoms are at index 1 in the atom37 representation.
    """
    atom37_list = _extract_atom37_list(candidates)
    ca_coords_list = []
    
    for atom37 in atom37_list:
        # CA is at index 1 in atom37 format
        ca_coords = atom37[:, 1, :].numpy()
        ca_coords_list.append(ca_coords)
    
    return ca_coords_list


def _extract_sequences(candidates: Any) -> List[str]:
    """Extract amino acid sequences from candidate inputs.
    
    Expects either:
    - List of strings (sequences)
    - Single string (sequence)
    """
    if isinstance(candidates, str):
        return [candidates]
    elif isinstance(candidates, list):
        sequences = []
        for item in candidates:
            if isinstance(item, str):
                sequences.append(item)
            else:
                raise TypeError(f"Expected string sequences, got {type(item)}")
        return sequences
    else:
        raise TypeError(f"Expected string or list of strings for sequence input, got {type(candidates)}")


def _write_atom37_to_pdb(
    coords: np.ndarray,
    cache_dir: Path,
    write_pdb_fn: Any,
) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        suffix=".pdb",
        dir=cache_dir,
        delete=False,
    )
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        # For ProteinMPNN compatibility: use placeholder aatype (all ALA) 
        # ProteinMPNN's parser requires residue names in PDB, but we only have coordinates
        # Using ALA (alanine, type 0) as placeholder - sequence doesn't matter for structure-based designability
        import numpy as np
        
        # Validate coords shape and content
        if coords.ndim != 3 or coords.shape[2] != 3:
            raise ValueError(f"Invalid coords shape for PDB writing: {coords.shape}, expected [N, 37, 3]")
        if coords.shape[1] != 37:
            raise ValueError(f"Invalid atom37 shape: {coords.shape[1]}, expected 37")
        
        n_residues = coords.shape[0]
        
        # Check for invalid coordinates (NaN, Inf, or all zeros)
        if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
            raise ValueError("Coordinates contain NaN or Inf values")
        if np.allclose(coords, 0):
            raise ValueError("Coordinates are all zeros - invalid structure")
        
        placeholder_aatype = np.zeros(n_residues, dtype=np.int32)  # All ALA residues
        
        write_pdb_fn(
            coords,
            str(tmp_path),
            aatype=placeholder_aatype,  # Add placeholder sequence for ProteinMPNN compatibility
            overwrite=True,
            no_indexing=True,
        )
        # Ensure file is flushed to disk before returning
        # This is important when files are used by subprocesses
        import os
        fd = os.open(str(tmp_path), os.O_RDONLY)
        try:
            os.fsync(fd)  # Force write to disk
        finally:
            os.close(fd)
    except Exception as exc:  # pragma: no cover - IO heavy
        tmp_path.unlink(missing_ok=True)
        # Include detailed error information for debugging
        error_details = f"Failed to serialize protein structure to PDB: {type(exc).__name__}: {str(exc)}"
        if hasattr(exc, '__cause__') and exc.__cause__:
            error_details += f" (caused by: {type(exc.__cause__).__name__}: {str(exc.__cause__)})"
        raise RuntimeError(error_details) from exc
    return tmp_path


def _get_write_pdb_fn():
    _setup_proteina_path()
    from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb

    return write_prot_to_pdb


def _load_fold_mappings(proteina_dir: str) -> tuple[Dict[int, List[str]], Dict[str, List[int]]]:
    """Load cached mapping between GearNet class indices and CATH codes."""
    mapping_path = Path(proteina_dir) / "proteina_additional_files" / "metric_factory" / "features" / "fold_class_mappings_C_selected_A_T_cath_codes.pth"
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Fold mapping file not found at {mapping_path}. "
            "Please ensure proteina_additional_files are downloaded."
        )
    raw_entries = torch.load(mapping_path, map_location="cpu")
    idx_to_codes: Dict[int, List[str]] = {}
    code_to_indices: Dict[str, List[int]] = defaultdict(list)

    for idx, codes in raw_entries:
        idx = int(idx)
        dedup_codes = sorted(set(str(code) for code in codes))
        idx_to_codes[idx] = dedup_codes
        for code in dedup_codes:
            code_to_indices[code].append(idx)

    for code in code_to_indices:
        code_to_indices[code] = sorted(set(code_to_indices[code]))

    return idx_to_codes, code_to_indices






class DesignabilityReward(RewardFunction):
    """
    Black-box reward built on the Proteina designability metric (scRMSD).

    The reward expects CA/atom37 coordinates (B, N, 37, 3) produced by the
    Proteina inference wrapper. For each structure we:
      1) save a temporary PDB file,
      2) run the expensive scRMSD pipeline (ProteinMPNN + ESMFold),
      3) convert the minimum scRMSD to a reward in (0, 1] via exp(-rmsd).

    Even though this reward is expensive, it matches the intended usage: treat
    the structure-to-designability evaluation as a black-box function.
    """

    def __init__(self, config: RewardConfig) -> None:
        super().__init__("designability", config.device)

        # Setup path before importing proteinfoundation
        _setup_proteina_path()
        
        try:
            from proteinfoundation.metrics.designability import scRMSD
            from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb
        except Exception as exc:  # pragma: no cover - heavy dep
            raise ImportError(
                "DesignabilityReward requires the proteinfoundation package. "
                "Please activate the Proteina environment before enabling this reward. "
                "Make sure the proteina directory is accessible."
            ) from exc

        self._scrmsd_fn = scRMSD
        self._write_pdb_fn = write_prot_to_pdb

        self.cache_dir = _resolve_cache_dir(config, "tmp_proteina_reward")
        
        # Create subdirectories required by scRMSD (ProteinMPNN and ESMFold)
        # scRMSD expects: tmp_path/seqs/ and tmp_path/esm/ subdirectories
        (self.cache_dir / "seqs").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "esm").mkdir(parents=True, exist_ok=True)
        
        # Store ProteinMPNN weights path if provided in config
        self.pmpnn_weights_path = getattr(config, 'pmpnn_weights_path', None)
        
        # Number of sequences to extract per protein for designability evaluation
        # Default is 8 to match inference.py behavior
        metadata = config.metadata or {}
        self.num_seq_per_target = int(metadata.get("num_seq_per_target", 8))

    def get_input_format(self) -> str:
        return "atom37"

    def _write_structure(self, coords: np.ndarray) -> Path:
        return _write_atom37_to_pdb(coords, self.cache_dir, self._write_pdb_fn)

    def _score_single(self, coords: np.ndarray) -> float:
        score_start = time.time()
        pdb_path = self._write_structure(coords)
        try:
            # Match inference.py exactly: use ret_min=False, no path_to_model_weights
            # Use per-sample tmp directory like inference.py
            import tempfile
            import os
            with tempfile.TemporaryDirectory() as tmp_dir:
                rmsd_values = self._scrmsd_fn(
                    str(pdb_path),
                    ret_min=False,
                    tmp_path=tmp_dir,
                    num_seq_per_target=self.num_seq_per_target,
                )
                best_rmsd = min(rmsd_values) if rmsd_values else float('inf')
        finally:
            pdb_path.unlink(missing_ok=True)

        if not np.isfinite(best_rmsd):
            raise RuntimeError(f"Invalid scRMSD score: {best_rmsd}")

        # Convert RMSD to reward: exp(-rmsd)
        # Lower RMSD = higher reward (better designability)
        reward = float(np.exp(-best_rmsd))
        score_time = time.time() - score_start
        print(f"[DesignabilityReward] Single evaluation: Time: {score_time:.2f}s | RMSD: {best_rmsd:.4f} | Reward: {reward:.4f}")
        return reward

    def _score_batch(self, coords_list: list[np.ndarray]) -> list[float]:
        """Score multiple proteins in batch for better efficiency."""
        batch_start_time = time.time()
        
        # Write all PDB files first
        pdb_write_start = time.time()
        pdb_paths = []
        try:
            for coords in coords_list:
                pdb_path = self._write_structure(coords)
                pdb_paths.append(pdb_path)
            pdb_write_time = time.time() - pdb_write_start
            
            # Import batch scRMSD if available, otherwise fall back to sequential
            try:
                from proteinfoundation.metrics.designability import scRMSD_batch
                # Use batch version if available
                scrmd_start = time.time()
                all_scores = scRMSD_batch(
                    [str(p) for p in pdb_paths],
                    tmp_path=str(self.cache_dir),
                    path_to_model_weights=self.pmpnn_weights_path,
                    num_seq_per_target=self.num_seq_per_target,
                )
                scrmd_time = time.time() - scrmd_start
                
                total_time = time.time() - batch_start_time
                mean_reward = np.mean(all_scores) if all_scores else 0.0
                print(f"[DesignabilityReward] Batch evaluation: {len(coords_list)} proteins | "
                      f"PDB write: {pdb_write_time:.2f}s | scRMSD: {scrmd_time:.2f}s | "
                      f"Total: {total_time:.2f}s | Rewards: {[f'{r:.4f}' for r in all_scores]} | Mean: {mean_reward:.4f}")
            except (ImportError, AttributeError):
                # Fall back to sequential processing (original behavior)
                # scRMSD returns RMSD values, so we need to convert to rewards
                all_scores = []
                for i, pdb_path in enumerate(pdb_paths):
                    score_start = time.time()
                    # Use ret_min=True to get minimum RMSD directly (more efficient)
                    best_rmsd = self._scrmsd_fn(
                        str(pdb_path),
                        ret_min=True,
                        tmp_path=str(self.cache_dir),
                        path_to_model_weights=self.pmpnn_weights_path,
                        num_seq_per_target=self.num_seq_per_target,
                    )
                    score_time = time.time() - score_start
                    if not np.isfinite(best_rmsd):
                        raise RuntimeError(f"Invalid scRMSD score: {best_rmsd}")
                    # Convert RMSD to reward: exp(-rmsd)
                    reward = float(np.exp(-best_rmsd))
                    all_scores.append(reward)
                    print(f"[DesignabilityReward] Protein {i+1}/{len(pdb_paths)}: "
                          f"Time: {score_time:.2f}s | RMSD: {best_rmsd:.4f} | Reward: {reward:.4f}")
                
                total_time = time.time() - batch_start_time
                mean_reward = np.mean(all_scores) if all_scores else 0.0
                print(f"[DesignabilityReward] Sequential batch total: {total_time:.2f}s | Mean reward: {mean_reward:.4f}")
            
            return all_scores
        finally:
            # Clean up all temporary PDB files
            for pdb_path in pdb_paths:
                pdb_path.unlink(missing_ok=True)

    def evaluate(
        self,
        candidates: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        rewards: list[float] = []

        atom37_list = _extract_atom37_list(candidates)
        if len(atom37_list) > 1:
            rewards = self._score_batch([coords.numpy() for coords in atom37_list])
        elif len(atom37_list) == 1:
            rewards = [self._score_single(atom37_list[0].numpy())]
        else:
            rewards = []

        return torch.tensor(rewards, device=self.device, dtype=torch.float32)






