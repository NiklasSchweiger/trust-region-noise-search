from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import math

from .molecule import MoleculeSamplingWrapper


class ProteinaWrapper(MoleculeSamplingWrapper):
    """Adaptor for Proteina backbones that exposes a simple noise → reward API.
    
    We need to do the following update:

    1. Add SDE sampling with a precomputed noise sequence.
    2. Always inject the noise sequence and the source noise, that comes for the solver.
    
    """



    def __init__(
        self,
        model: Any,
        *,
        model_cfg: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        default_n_residues: int = 100,
        disable_tqdm: bool = True,
        verbose: bool = False,
        use_zero_noise: bool = False,
        deterministic_seed: Optional[int] = None,
        noise_seed_offset: int = 0,
        precomputed_noise: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(model, device=device)
        self.model_cfg = model_cfg or {}
        self.default_n_residues = int(default_n_residues)
        self.underlying_model = model
        self.underlying_model.eval()
        self.underlying_model.to(device)
        self.disable_tqdm = bool(disable_tqdm)
        self.verbose = bool(verbose)
        
        # Get max_nsamples from config to respect Proteina's internal batch size limits
        self.max_nsamples = int(self.model_cfg.get("max_nsamples", 10))

        # Noise control for deterministic behavior
        self.use_zero_noise = bool(use_zero_noise)
        self.deterministic_seed = deterministic_seed
        self.noise_seed_offset = int(noise_seed_offset)
        self._noise_call_count = 0  # Track calls for deterministic seeding

        # Precomputed noise for reproducible experiments
        self.precomputed_noise = precomputed_noise
        if self.precomputed_noise is not None:
            self.precomputed_noise = self.precomputed_noise.to(device, dtype=torch.float32)

        # Precomputed SDE noise sequence for deterministic stochastic sampling
        self.precomputed_sde_noise = None

        # Disable tqdm if requested (for cleaner benchmark output)
        if self.disable_tqdm:
            import os
            os.environ["DISABLE_TQDM"] = "1"

    # ------------------------------------------------------------------ #
    # Noise helpers
    # ------------------------------------------------------------------ #
    def get_noise_latent_shape(self) -> Tuple[int, ...]:
        return (self.default_n_residues, 3)

    def get_sde_noise(self, shape: Tuple[int, ...], step_idx: int = 0) -> torch.Tensor:
        """Get deterministic SDE noise for integration steps.

        Uses a fixed sequence that cycles through precomputed noise,
        ensuring the same noise pattern for all optimization runs.
        """
        if self.precomputed_sde_noise is None:
            # Fallback to random noise if no precomputed sequence
            return torch.randn(shape, dtype=torch.float32, device=self.device)

        # Cycle through the precomputed sequence
        batch_size, n_residues, dim = shape
        seq_idx = step_idx % len(self.precomputed_sde_noise)
        noise = self.precomputed_sde_noise[seq_idx]

        # Ensure shape matches requested
        if noise.shape[0] != batch_size or noise.shape[1] != n_residues:
            # Repeat or slice as needed
            if noise.shape[0] < batch_size:
                noise = noise.repeat(batch_size // noise.shape[0] + 1, 1, 1)[:batch_size]
            elif noise.shape[0] > batch_size:
                noise = noise[:batch_size]

            if noise.shape[1] != n_residues:
                # For different n_residues, we'll need to interpolate or truncate
                # For simplicity, truncate/extend
                if noise.shape[1] > n_residues:
                    noise = noise[:, :n_residues]
                else:
                    # Pad with zeros
                    padding = torch.zeros(batch_size, n_residues - noise.shape[1], dim,
                                        dtype=noise.dtype, device=noise.device)
                    noise = torch.cat([noise, padding], dim=1)

        return noise.to(self.device, dtype=torch.float32)

    def sample_latents(self, batch_size: int = 1, *, n_residues: Optional[int] = None) -> torch.Tensor:
        n = int(n_residues or self.default_n_residues)

        # Use precomputed noise if available
        if self.precomputed_noise is not None:
            if self.precomputed_noise.shape[0] < batch_size:
                raise ValueError(f"Precomputed noise batch size {self.precomputed_noise.shape[0]} is smaller than requested batch size {batch_size}")
            if self.precomputed_noise.shape[1] != n:
                raise ValueError(f"Precomputed noise has {self.precomputed_noise.shape[1]} residues but requested {n} residues")

            # Return subset of precomputed noise for this batch
            noise = self.precomputed_noise[:batch_size].to(self.device, dtype=torch.float32)
            # Apply the same masking and centering as sample_reference would do
            return self.underlying_model.fm._mask_and_zero_com(noise)

        # For proteins, we should use the model's own noise distribution
        # Don't use zero noise or deterministic seeds for optimization - let TRS explore properly
        return self.underlying_model.fm.sample_reference(
            n=n,
            shape=(batch_size,),
            device=self.device,
            dtype=torch.float32,
        )

    # Inference
    # ------------------------------------------------------------------ #
    def forward(
        self,
        batch_size: int = 1,
        initial_noise: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # CRITICAL DEBUG: Print at the VERY FIRST LINE to confirm function is entered
        # sys.stdout.write("[PROTEINA WRAPPER] >>>>> forward() ENTERED <<<<<\n")
        # sys.stdout.flush()
        
        # UNCONDITIONAL DEBUG: Always print with flush to ensure it appears
        # print("=" * 80, flush=True)
        # print("[PROTEINA WRAPPER] forward() CALLED!", flush=True)
        # print(f"  batch_size: {batch_size}", flush=True)
        # print(f"  initial_noise: {'SET' if initial_noise is not None else 'None'}", flush=True)
        # print(f"  kwargs keys: {list(kwargs.keys())}", flush=True)
        # print(f"  kwargs.get('sampling_mode'): {kwargs.get('sampling_mode', 'NOT FOUND')}", flush=True)
        # print(f"  kwargs.get('sc_scale_noise'): {kwargs.get('sc_scale_noise', 'NOT FOUND')}", flush=True)
        # print(f"  kwargs.get('sc_scale_score'): {kwargs.get('sc_scale_score', 'NOT FOUND')}", flush=True)
        # print(f"  kwargs.get('n_residues'): {kwargs.get('n_residues', 'NOT FOUND')}", flush=True)
        # print("=" * 80, flush=True)
        
        grad_enabled = kwargs.get("differentiable", False)
        
        with torch.set_grad_enabled(grad_enabled):
            n_residues = int(kwargs.get("n_residues", kwargs.get("length", self.default_n_residues)))
        batch_size = int(batch_size)
        if initial_noise is None:
            # For random generation, let the model use its own noise
            noise = None
        else:
            # For optimization, use the provided noise
            noise = initial_noise.to(self.device, dtype=torch.float32)
            if noise.dim() != 3:
                raise ValueError(f"Expected noise tensor [B, N, 3], got {noise.shape}")
            n_residues = noise.shape[1]

        sampling_kwargs = self._resolve_sampling_kwargs(kwargs, batch_size=batch_size)

        # Check if we need to chunk based on max_nsamples
        needs_chunking = batch_size > self.max_nsamples

        # Only override sample_reference if we have custom noise AND not chunking
        # (If chunking, we'll handle sample_reference per chunk)
        original_sample_ref = None
        if noise is not None and not needs_chunking:
            original_sample_ref = self.underlying_model.fm.sample_reference

            def custom_sample_ref(n: int, shape, device=None, mask=None, dtype=None):
                if n != n_residues:
                    raise ValueError(f"Requested n={n} but noise was prepared for {n_residues} residues.")
                expected_batch = shape[0] if isinstance(shape, tuple) and shape else 1
                if expected_batch != noise.shape[0]:
                    raise ValueError(f"Requested batch {expected_batch} but noise batch is {noise.shape[0]}.")
                return self.underlying_model.fm._mask_and_zero_com(noise, mask)

            self.underlying_model.fm.sample_reference = custom_sample_ref

        try:
            # Extract cath_code explicitly since it's a required positional argument
            cath_code = sampling_kwargs.pop("cath_code", None)

            # Extract motif information if present
            motif_seq_mask = kwargs.get("motif_seq_mask")
            motif_structure = kwargs.get("motif_structure")

            # Pass motif info to generate() if available
            motif_kwargs = {}
            if motif_seq_mask is not None and motif_structure is not None:
                # Ensure tensors are on correct device and have correct batch size
                if isinstance(motif_seq_mask, torch.Tensor):
                    motif_seq_mask = motif_seq_mask.to(self.device)
                    # Expand to batch_size if needed
                    if motif_seq_mask.shape[0] == 1 and batch_size > 1:
                        motif_seq_mask = motif_seq_mask.repeat(batch_size, 1)
                    elif motif_seq_mask.shape[0] != batch_size:
                        motif_seq_mask = motif_seq_mask[:batch_size]

                if isinstance(motif_structure, torch.Tensor):
                    motif_structure = motif_structure.to(self.device)
                    # Expand to batch_size if needed
                    if motif_structure.shape[0] == 1 and batch_size > 1:
                        motif_structure = motif_structure.repeat(batch_size, 1, 1)
                    elif motif_structure.shape[0] != batch_size:
                        motif_structure = motif_structure[:batch_size]

                motif_kwargs["fixed_sequence_mask"] = motif_seq_mask
                motif_kwargs["x_motif"] = motif_structure

            # UNCONDITIONAL DEBUG: Always print with flush to ensure it appears
            # sys.stdout.flush()
            # print("=" * 80, flush=True)
            # print("[PROTEINA WRAPPER] About to call underlying_model.generate() with:", flush=True)
            # print(f"  nsamples: {batch_size}, n: {n_residues}, cath_code: {cath_code}", flush=True)
            # print(f"  sampling_kwargs keys: {list(sampling_kwargs.keys())}", flush=True)
            # print(f"  sampling_kwargs['sampling_mode']: {sampling_kwargs.get('sampling_mode', 'NOT IN KWARGS')}", flush=True)
            # print(f"  sampling_kwargs['sc_scale_noise']: {sampling_kwargs.get('sc_scale_noise', 'NOT IN KWARGS')} (type: {type(sampling_kwargs.get('sc_scale_noise', None))})", flush=True)
            # print(f"  sampling_kwargs['sc_scale_score']: {sampling_kwargs.get('sc_scale_score', 'NOT IN KWARGS')} (type: {type(sampling_kwargs.get('sc_scale_score', None))})", flush=True)
            # print(f"  sampling_kwargs['gt_mode']: {sampling_kwargs.get('gt_mode', 'NOT IN KWARGS')}", flush=True)
            # print("=" * 80, flush=True)
            
            # Respect max_nsamples limit by splitting large batches into chunks
            # This prevents memory issues and respects Proteina's internal batch size limits
            if batch_size > self.max_nsamples:
                # Split batch into chunks of max_nsamples
                results = []
                for i in range(0, batch_size, self.max_nsamples):
                    chunk_size = min(self.max_nsamples, batch_size - i)
                    
                    # Temporarily override sample_reference for this chunk if we have custom noise
                    chunk_noise = noise[i:i + chunk_size] if noise is not None else None
                    chunk_original_sample_ref = None
                    if chunk_noise is not None:
                        chunk_original_sample_ref = self.underlying_model.fm.sample_reference
                        def chunk_custom_sample_ref(n: int, shape, device=None, mask=None, dtype=None):
                            if n != n_residues:
                                raise ValueError(f"Requested n={n} but noise was prepared for {n_residues} residues.")
                            expected_batch = shape[0] if isinstance(shape, tuple) and shape else 1
                            if expected_batch != chunk_noise.shape[0]:
                                raise ValueError(f"Requested batch {expected_batch} but noise batch is {chunk_noise.shape[0]}.")
                            return self.underlying_model.fm._mask_and_zero_com(chunk_noise, mask)
                        self.underlying_model.fm.sample_reference = chunk_custom_sample_ref
                    
                    try:
                        # Handle cath_code chunking
                        chunk_cath_code = None
                        if cath_code is not None:
                            if isinstance(cath_code, list):
                                chunk_cath_code = cath_code[i:i + chunk_size]
                            else:
                                chunk_cath_code = cath_code
                        
                        # Handle motif kwargs chunking
                        chunk_motif_kwargs = {}
                        if motif_kwargs:
                            for key, value in motif_kwargs.items():
                                if isinstance(value, torch.Tensor) and value.shape[0] == batch_size:
                                    chunk_motif_kwargs[key] = value[i:i + chunk_size]
                                else:
                                    chunk_motif_kwargs[key] = value
                        
                        # Generate chunk
                        chunk_raw = self.underlying_model.generate(
                            nsamples=chunk_size,
                            n=n_residues,
                            cath_code=chunk_cath_code,
                            **sampling_kwargs,
                            **chunk_motif_kwargs,
                        )
                        results.append(chunk_raw)
                    finally:
                        # Restore original sample_reference if it was overridden
                        if chunk_original_sample_ref is not None:
                            self.underlying_model.fm.sample_reference = chunk_original_sample_ref
                
                # Concatenate results
                raw = {k: torch.cat([r[k] for r in results], dim=0) for k in results[0].keys()}
            else:
                # Batch size is within limit, process normally
                raw = self.underlying_model.generate(
                    nsamples=batch_size,
                    n=n_residues,
                    cath_code=cath_code,
                    **sampling_kwargs,
                    **motif_kwargs,
                )

            atom37 = self.underlying_model.samples_to_atom37(raw)
            return atom37

        finally:
            # Restore original sample_reference if it was overridden
            if original_sample_ref is not None:
                self.underlying_model.fm.sample_reference = original_sample_ref


    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #
    def _resolve_sampling_kwargs(self, user_kwargs: Dict[str, Any], batch_size: int = 1) -> Dict[str, Any]:
        # Use instance config values as defaults, overridden by user_kwargs
        # UNCONDITIONAL DEBUG: Always print with flush
        # sys.stdout.flush()
        # print(f"[PROTEINA DEBUG] _resolve_sampling_kwargs CALLED!", flush=True)
        # print(f"  - user_kwargs keys: {list(user_kwargs.keys())}", flush=True)
        # print(f"  - user_kwargs.get('sampling_mode'): {user_kwargs.get('sampling_mode', 'NOT FOUND')}", flush=True)
        # print(f"  - user_kwargs.get('sc_scale_noise'): {user_kwargs.get('sc_scale_noise', 'NOT FOUND')}", flush=True)
        # print(f"  - self.model_cfg.get('sampling_mode'): {self.model_cfg.get('sampling_mode', 'NOT FOUND')}", flush=True)
        # print(f"  - self.model_cfg.get('sc_scale_noise'): {self.model_cfg.get('sc_scale_noise', 'NOT FOUND')}", flush=True)
        
        cfg = {
            "dt": float(user_kwargs.get("dt", self.model_cfg.get("dt", 0.0025))),
            "self_cond": bool(user_kwargs.get("self_cond", self.model_cfg.get("self_cond", False))),
            "guidance_weight": float(user_kwargs.get("guidance_weight", self.model_cfg.get("guidance_weight", 1.0))),
            "autoguidance_ratio": float(user_kwargs.get("autoguidance_ratio", self.model_cfg.get("autoguidance_ratio", 1.0))),
            "dtype": torch.float32,
            "schedule_mode": user_kwargs.get("schedule_mode", self.model_cfg.get("schedule_mode", "log")),
            "schedule_p": float(user_kwargs.get("schedule_p", self.model_cfg.get("schedule_p", 2.0))),
            "sampling_mode": user_kwargs.get("sampling_mode", self.model_cfg.get("sampling_mode", "sc")),
            "sc_scale_noise": float(user_kwargs.get("sc_scale_noise", self.model_cfg.get("sc_scale_noise", 0.4))),
            "sc_scale_score": float(user_kwargs.get("sc_scale_score", self.model_cfg.get("sc_scale_score", 1.0))),
            "gt_mode": user_kwargs.get("gt_mode", self.model_cfg.get("gt_mode", "1/t")),
            "gt_p": float(user_kwargs.get("gt_p", self.model_cfg.get("gt_p", 1.0))),
            "gt_clamp_val": user_kwargs.get("gt_clamp_val", self.model_cfg.get("gt_clamp_val")),
        }
        
        # DEBUG: Print resolved values with flush
        # print(f"[PROTEINA DEBUG] Resolved config:", flush=True)
        # print(f"  - sampling_mode: {cfg['sampling_mode']}", flush=True)
        # print(f"  - sc_scale_noise: {cfg['sc_scale_noise']}", flush=True)
        print(f"  - sc_scale_score: {cfg['sc_scale_score']}")
        
        # Handle cath_code: convert single fold code to List[List[str]] format
        # Proteina expects List[List[str]] where each inner list has one code per sample
        # For unconditional generation, pass None
        cath_code_raw = user_kwargs.get("cath_code")
        if cath_code_raw is not None:
            if isinstance(cath_code_raw, str):
                # Single fold code: convert to list of lists (repeat for each sample)
                cfg["cath_code"] = [[cath_code_raw] for _ in range(batch_size)]
            elif isinstance(cath_code_raw, list) and len(cath_code_raw) > 0:
                if isinstance(cath_code_raw[0], str):
                    # List of strings: wrap each in a list and pad/repeat to match batch_size
                    codes = list(cath_code_raw)
                    if len(codes) < batch_size:
                        # Repeat last code to fill batch
                        codes.extend([codes[-1]] * (batch_size - len(codes)))
                    cfg["cath_code"] = [[code] for code in codes[:batch_size]]
                elif isinstance(cath_code_raw[0], list):
                    # Already List[List[str]] format, just ensure correct length
                    codes = list(cath_code_raw)
                    if len(codes) < batch_size:
                        codes.extend([codes[-1]] * (batch_size - len(codes)))
                    cfg["cath_code"] = codes[:batch_size]
        else:
            # Unconditional generation: pass None (generate() will handle it)
            cfg["cath_code"] = None
        
        return cfg


# ============================================================================
# Factory function for Proteina models
# ============================================================================

def make_proteina_from_cfg(cfg: dict, device: str = "cuda"):
    """Factory that builds a Proteina model from config.
    
    Expected keys in cfg:
      - proteina.ckpt_path: path to checkpoint directory (required)
      - proteina.data_path: path to data directory (optional, for fold-conditioned generation)
      - proteina.lora.use: whether to use LoRA (optional, default: False)
      - proteina.lora.r: LoRA rank (optional, default: 16)
      - proteina.lora.lora_alpha: LoRA alpha (optional, default: 32.0)
      - proteina.lora.lora_dropout: LoRA dropout (optional, default: 0.0)
      - proteina.lora.train_bias: LoRA bias training (optional, default: "none")
      - proteina.n_residues: default number of residues (optional, default: 100)
      - proteina.sampling_mode: sampling mode (optional, default: "vf")
      - proteina.sc_scale_noise: self-conditioning noise scale (optional, default: 0.4)
      - proteina.sc_scale_score: self-conditioning score scale (optional, default: 1.0)
      - proteina.gt_mode: guidance target mode (optional, default: "1/t")
      - proteina.gt_p: guidance target parameter (optional, default: 1.0)
      - proteina.gt_clamp_val: guidance target clamp value (optional)
      - proteina.schedule_mode: schedule mode (optional, default: "log")
      - proteina.schedule_p: schedule parameter (optional, default: 2.0)
      - proteina.fold_cond: whether to use fold conditioning (optional, default: False)
      - proteina.guidance_weight: guidance weight (optional, default: 1.0)
      - proteina.autoguidance_ratio: autoguidance ratio (optional, default: 0.0)
      - proteina.self_cond: whether to use self-conditioning (optional, default: False)
      - proteina.use_zero_noise: whether to use zero noise for deterministic behavior (optional, default: False)
      - proteina.deterministic_seed: seed for deterministic noise generation (optional, default: None)
      - proteina.noise_seed_offset: offset for deterministic seed sequence (optional, default: 0)
      - proteina.precomputed_noise_path: path to precomputed noise tensor file (.pt) (optional, default: None)
      - proteina.precomputed_noise: precomputed noise tensor (optional, default: None)
      - proteina.precomputed_sde_noise_path: path to precomputed SDE noise sequence file (.pt) (optional, default: None)
      - proteina.precomputed_sde_noise: precomputed SDE noise tensor/sequence (optional, default: None)
      - proteina.sde_noise_seed: seed for generating reproducible SDE noise sequence (optional, default: None)
      - proteina.sde_noise_seq_length: length of SDE noise sequence to generate (optional, default: 1000)
    """
    import os
    import tempfile
    from omegaconf import OmegaConf
    
    # Setup application paths
    current_file = os.path.abspath(__file__)
    noise_opt_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    proteina_dir = os.path.join(noise_opt_dir, "proteina")
    if os.path.exists(proteina_dir) and proteina_dir not in sys.path:
        sys.path.insert(0, proteina_dir)
    
    from proteinfoundation.proteinflow.proteina import Proteina
    import loralib as lora
    from proteinfoundation.utils.lora_utils import replace_lora_layers
    
    model_cfg = cfg.get("proteina") or {}
    if isinstance(model_cfg, dict):
        pass
    else:
        model_cfg = OmegaConf.to_container(model_cfg, resolve=True) if hasattr(model_cfg, '__dict__') else dict(model_cfg)
    
    ckpt_path = model_cfg.get("ckpt_path")
    ckpt_name = model_cfg.get("ckpt_name")
    if ckpt_path is None:
        raise ValueError("proteina.ckpt_path is required")

    # Handle both directory + filename format (like inference.py) or full path
    if ckpt_name:
        # Directory + filename format
        ckpt_file = os.path.join(ckpt_path, ckpt_name)
    else:
        # Assume ckpt_path is the full path to checkpoint file
        ckpt_file = ckpt_path

    # Validate checkpoint path
    if "/path/to/your/" in str(ckpt_file) or "/path/to/" in str(ckpt_file):
        raise ValueError(
            f"Invalid checkpoint path: '{ckpt_file}'. "
            "Please provide the actual path to your Proteina checkpoint. "
            "Example: proteina.ckpt_path=/path/to/dir, proteina.ckpt_name=checkpoint.ckpt"
        )

    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(
            f"Checkpoint file not found: '{ckpt_file}'. "
            "Please ensure the path is correct and exists."
        )
    
    # Load .env file from proteina directory if it exists
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(proteina_dir, ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path, override=False)
            print(f"[INFO] Loaded .env from {env_path}")
    except ImportError:
        pass
    
    # Set DATA_PATH environment variable if not already set
    if "DATA_PATH" not in os.environ:
        config_data_path = model_cfg.get("data_path")
        if config_data_path:
            os.environ["DATA_PATH"] = str(config_data_path)
            print(f"[INFO] Using DATA_PATH from config: {config_data_path}")
        else:
            default_data_path = os.path.join(tempfile.gettempdir(), "proteina_data_path")
            os.makedirs(default_data_path, exist_ok=True)
            os.environ["DATA_PATH"] = default_data_path
            print(f"[INFO] DATA_PATH not set, using temporary default: {default_data_path}")
            print(
                "[INFO] For fold-conditioned generation, create proteina/.env with DATA_PATH=/path/to/data or set proteina.data_path in config"
            )
    
    print(f"Loading Proteina model from {ckpt_file}...")

    lora_cfg = model_cfg.get("lora", {})
    use_lora = lora_cfg.get("use", False)

    if not use_lora:
        model = Proteina.load_from_checkpoint(ckpt_file)
    else:
        model = Proteina.load_from_checkpoint(ckpt_file, strict=False)
        print("Re-create LoRA layers and reload the weights now")
        replace_lora_layers(
            model,
            lora_cfg.get("r", 16),
            lora_cfg.get("lora_alpha", 32.0),
            lora_cfg.get("lora_dropout", 0.0),
        )
        lora.mark_only_lora_as_trainable(model, bias=lora_cfg.get("train_bias", "none"))
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])

    # Configure inference - create nested config structure like inference.py expects
    inf_cfg = OmegaConf.create({
        # Top-level parameters like inference.py
        "dt": model_cfg.get("dt", 0.0025),
        "self_cond": model_cfg.get("self_cond", False),
        "guidance_weight": model_cfg.get("guidance_weight", 1.0),
        "autoguidance_ratio": model_cfg.get("autoguidance_ratio", 1.0),
        "autoguidance_ckpt_path": model_cfg.get("autoguidance_ckpt_path"),
        "fold_cond": model_cfg.get("fold_cond", False),
        "cath_code_level": model_cfg.get("cath_code_level", "T"),
        "len_cath_code_path": model_cfg.get("len_cath_code_path"),
        "seed": model_cfg.get("seed", 5),

        # Nested structures like inference.py
        "sampling_caflow": {
            "sampling_mode": model_cfg.get("sampling_mode", "sc"),
            "sc_scale_noise": model_cfg.get("sc_scale_noise", 0.4),
            "sc_scale_score": model_cfg.get("sc_scale_score", 1.0),
            "gt_mode": model_cfg.get("gt_mode", "1/t"),
            "gt_p": model_cfg.get("gt_p", 1.0),
            "gt_clamp_val": model_cfg.get("gt_clamp_val"),
        },
        "schedule": {
            "schedule_mode": model_cfg.get("schedule_mode", "log"),
            "schedule_p": model_cfg.get("schedule_p", 2.0),
        },

        # Additional parameters that might be needed
        "nres_lens": model_cfg.get("nres_lens", [100]),
        "nsamples_per_len": model_cfg.get("nsamples_per_len", 1),
        "max_nsamples": model_cfg.get("max_nsamples", 10),
        "designability_seqs_per_struct": model_cfg.get("designability_seqs_per_struct", 8),
    })

    # Load precomputed noise if specified
    precomputed_noise = None
    precomputed_noise_path = model_cfg.get("precomputed_noise_path")
    if precomputed_noise_path is not None:
        if not os.path.exists(precomputed_noise_path):
            raise FileNotFoundError(f"Precomputed noise file not found: {precomputed_noise_path}")
        print(f"[INFO] Loading precomputed noise from {precomputed_noise_path}")
        precomputed_noise = torch.load(precomputed_noise_path, map_location="cpu")
        print(f"[INFO] Loaded precomputed noise with shape {precomputed_noise.shape}")
    elif model_cfg.get("precomputed_noise") is not None:
        # Direct tensor provided
        precomputed_noise = model_cfg.get("precomputed_noise")
        print(f"[INFO] Using provided precomputed noise with shape {precomputed_noise.shape}")

    # Generate or load precomputed SDE noise sequence
    precomputed_sde_noise = None
    sde_noise_seed = model_cfg.get("sde_noise_seed")
    sde_noise_seq_length = model_cfg.get("sde_noise_seq_length", 1000)

    # Priority: direct tensor > seed generation > file loading
    if model_cfg.get("precomputed_sde_noise") is not None:
        # Direct tensor/list provided
        loaded_noise = model_cfg.get("precomputed_sde_noise")
        if isinstance(loaded_noise, list):
            precomputed_sde_noise = loaded_noise
        elif isinstance(loaded_noise, torch.Tensor):
            if loaded_noise.dim() == 3:
                precomputed_sde_noise = [loaded_noise]  # Single tensor, repeat for all steps
            elif loaded_noise.dim() == 4:
                precomputed_sde_noise = [loaded_noise[i] for i in range(loaded_noise.shape[0])]
            else:
                raise ValueError(f"Invalid SDE noise tensor shape {loaded_noise.shape}")
        print(f"[INFO] Using provided precomputed SDE noise sequence with {len(precomputed_sde_noise)} steps")

    elif sde_noise_seed is not None:
        # Generate reproducible sequence using seed
        print(f"[INFO] Generating reproducible SDE noise sequence (seed={sde_noise_seed}, length={sde_noise_seq_length})")

        # Use the model's n_residues for noise generation
        n_residues = model_cfg.get("n_residues", 100)
        scale_ref = 1.0  # Same as R3NFlowMatcher

        # Generate sequence with fixed seed
        original_seed = torch.initial_seed()
        try:
            torch.manual_seed(sde_noise_seed)
            precomputed_sde_noise = []
            for _ in range(sde_noise_seq_length):
                # Generate noise with shape [1, n_residues, 3] - will be adapted to batch size later
                noise = torch.randn(1, n_residues, 3, dtype=torch.float32) * scale_ref
                precomputed_sde_noise.append(noise)
        finally:
            # Restore original random state
            torch.manual_seed(original_seed)

        print(f"[INFO] Generated SDE noise sequence with {len(precomputed_sde_noise)} steps")

    elif model_cfg.get("precomputed_sde_noise_path") is not None:
        # Fallback: load from file
        precomputed_sde_noise_path = model_cfg.get("precomputed_sde_noise_path")
        if not os.path.exists(precomputed_sde_noise_path):
            raise FileNotFoundError(f"Precomputed SDE noise file not found: {precomputed_sde_noise_path}")
        print(f"[INFO] Loading precomputed SDE noise sequence from {precomputed_sde_noise_path}")
        loaded_noise = torch.load(precomputed_sde_noise_path, map_location="cpu")
        print(f"[INFO] Loaded precomputed SDE noise with shape {loaded_noise.shape}")

        # Convert to list of tensors for sequential access
        if loaded_noise.dim() == 4:
            # Shape: [seq_len, batch_size, n_residues, 3] - sequence of noise tensors
            precomputed_sde_noise = [loaded_noise[i] for i in range(loaded_noise.shape[0])]
        elif loaded_noise.dim() == 3:
            # Single noise tensor - use repeatedly for all steps
            precomputed_sde_noise = [loaded_noise]
        else:
            raise ValueError(f"Invalid SDE noise shape {loaded_noise.shape}, expected [seq_len, batch_size, n_residues, 3] or [batch_size, n_residues, 3]")

    # Load autoguidance network if needed
    nn_ag = None
    if (
        inf_cfg.get("autoguidance_ratio", 0.0) > 0
        and inf_cfg.get("guidance_weight", 1.0) != 1.0
    ):
        autoguidance_ckpt_path = model_cfg.get("autoguidance_ckpt_path")
        if autoguidance_ckpt_path is None:
            print(f"⚠️  [WARNING] autoguidance_ratio > 0 but autoguidance_ckpt_path is not set. Autoguidance will not be used.")
        else:
            if not os.path.exists(autoguidance_ckpt_path):
                raise FileNotFoundError(f"Autoguidance checkpoint not found: {autoguidance_ckpt_path}")
            print(f"🔧 [ProteinaWrapper] Loading autoguidance network from {autoguidance_ckpt_path}")
            model_ag = Proteina.load_from_checkpoint(autoguidance_ckpt_path)
            nn_ag = model_ag.nn
            print(f"[ProteinaWrapper] Autoguidance network loaded successfully")
    
    print(f"🔧 [ProteinaWrapper] Configuring model with:")
    print(f"  sampling_mode: {inf_cfg.sampling_caflow.sampling_mode}")
    print(f"  guidance_weight: {inf_cfg.guidance_weight}")
    print(f"  autoguidance_ratio: {inf_cfg.autoguidance_ratio}")
    if precomputed_noise is not None:
        print(f"  precomputed_noise: {precomputed_noise.shape}")
    model.configure_inference(inf_cfg, nn_ag=nn_ag)

    # DEBUG: Print what's in model_cfg before creating wrapper
    # print("=" * 80)
    # print("[PROTEINA FACTORY] model_cfg contents:")
    # print(f"  - sampling_mode: {model_cfg.get('sampling_mode', 'NOT FOUND')}")
    # print(f"  - sc_scale_noise: {model_cfg.get('sc_scale_noise', 'NOT FOUND')}")
    # print(f"  - sc_scale_score: {model_cfg.get('sc_scale_score', 'NOT FOUND')}")
    # print(f"  - All model_cfg keys: {list(model_cfg.keys())}")
    # print("=" * 80)
    
    wrapper = ProteinaWrapper(
        model,
        model_cfg=model_cfg,
        device=device,
        default_n_residues=int(model_cfg.get("n_residues", 100)),
        disable_tqdm=False,  # Enable tqdm progress bars
        verbose=False,
        use_zero_noise=model_cfg.get("use_zero_noise", False),
        deterministic_seed=model_cfg.get("deterministic_seed"),
        noise_seed_offset=model_cfg.get("noise_seed_offset", 0),
        precomputed_noise=precomputed_noise,
    )

    # Set precomputed SDE noise on the wrapper
    if precomputed_sde_noise is not None:
        wrapper.precomputed_sde_noise = precomputed_sde_noise
        print(f"[INFO] Set precomputed SDE noise sequence with {len(precomputed_sde_noise)} steps")
    return wrapper
