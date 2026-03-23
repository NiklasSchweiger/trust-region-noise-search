# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Union

import einops
import numpy as np
import torch
from jaxtyping import Float
from loguru import logger
from torch import Tensor
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers import logging as hf_logging
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein
from transformers.models.esm.openfold_utils.protein import to_pdb

from proteinfoundation.utils.align_utils.align_utils import kabsch_align_ind
from proteinfoundation.utils.ff_utils.pdb_utils import from_pdb_string

hf_logging.set_verbosity_error()


def pdb_name_from_path(pdb_file_path):
    return pdb_file_path.strip(os.sep).split(os.sep)[-1][
        :-4
    ]  # Name of the pdb file without ".pdb" extension


# ProteinMPNN
## ## ## ## ## ## ## ## ## ## ## ##


def extract_gen_seqs(path_to_file: str) -> List[str]:
    """
    Extracts sequences from ProteinMPNN generation files.

    Args:
        path_to_file: Path to file with pmpnn output.

    Returns:
        List of sequences produced by pmpnn.
    """
    seqs = []
    with open(path_to_file, "r") as f:
        first = True  # Assuming first sequence is not a generation
        for line in f:
            if not line.startswith(">"):
                if first:
                    first = False
                    continue
                else:
                    seqs.append(line.strip())
    return seqs


def run_proteinmpnn(
    pdb_file_path: str,
    out_dir_root: str,
    sampling_temp: float = 0.1,
    num_seq_per_target: int = 8,
    seed: Optional[int] = None,
    ca_only: bool = True,
    verbose: bool = False,
    path_to_model_weights: Optional[str] = None,
) -> List[str]:
    """
    Just an interfact to ProteinMPNN.

    Args:
        pdb_file_path: path to PDB file
        out_dir_root: Path used to store produced sequences
        sampling_temp: Sampling temperature for ProteinMPNN
        num_seq_per_target: Number of sequences produced per target provided
        seed: Random seed used for sampling
        ca_only: Whether to only use alpha carbons
        verbose: Print stuff or not

    Returns:
        List of sequences (strings)
    """
    name = pdb_name_from_path(pdb_file_path)

    python_exec = os.environ.get("PYTHON_EXEC")
    if python_exec is None:
        python_exec = "python"

    # Find ProteinMPNN script - try relative path first, then absolute
    current_dir = os.getcwd()
    proteina_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    mpnn_script_rel = os.path.join(proteina_dir, "ProteinMPNN", "protein_mpnn_run.py")
    mpnn_script_abs = os.path.abspath(mpnn_script_rel)
    
    if not os.path.exists(mpnn_script_abs):
        # Try relative to current directory
        mpnn_script_abs = os.path.abspath("./ProteinMPNN/protein_mpnn_run.py")
        if not os.path.exists(mpnn_script_abs):
            raise FileNotFoundError(
                f"Could not find ProteinMPNN script. Tried: {mpnn_script_rel} and ./ProteinMPNN/protein_mpnn_run.py"
            )
    
    # Convert out_dir_root to absolute path to avoid path resolution issues
    # when running from proteina_dir as working directory
    if not os.path.isabs(out_dir_root):
        # If relative, resolve relative to current working directory (before we cd to proteina_dir)
        out_dir_root = os.path.abspath(out_dir_root)
    
    # Also convert PDB file path to absolute to avoid issues when running from different directory
    if not os.path.isabs(pdb_file_path):
        pdb_file_path = os.path.abspath(pdb_file_path)
    
    # Ensure output directory exists
    os.makedirs(os.path.join(out_dir_root, "seqs"), exist_ok=True)
    
    # Build command as list for subprocess
    cmd_parts = [
        python_exec,
        mpnn_script_abs,
        "--pdb_path", pdb_file_path,
        "--pdb_path_chains", "A",
        "--out_folder", out_dir_root,
        "--num_seq_per_target", str(num_seq_per_target),
        "--sampling_temp", str(sampling_temp),
        "--batch_size", "1",
        "--suppress_print", str(0 if verbose else 1),
    ]
    
    if ca_only:
        cmd_parts.append("--ca_only")
    if seed is not None:
        cmd_parts.extend(["--seed", str(seed)])
    if path_to_model_weights:
        cmd_parts.extend(["--path_to_model_weights", path_to_model_weights])
    else:
        # Check if default weights location exists, provide helpful error if not
        default_weights_dir = os.path.join(proteina_dir, "ProteinMPNN", "ca_model_weights" if ca_only else "vanilla_model_weights")
        default_model_file = os.path.join(default_weights_dir, "v_48_020.pt")
        if not os.path.exists(default_model_file):
            error_msg = (
                f"ProteinMPNN model weights not found at: {default_model_file}\n"
                f"Please download the weights by running:\n"
                f"  bash {os.path.join(proteina_dir, 'script_utils', 'download_pmpnn_weghts.sh')}\n"
                f"Or specify the path via reward_function.pmpnn_weights_path in your config."
            )
            raise FileNotFoundError(error_msg)
    
    # Run command and capture output for error reporting
    import subprocess
    import time
    
    # Run from proteina directory so ProteinMPNN can find its files
    result = subprocess.run(
        cmd_parts,
        capture_output=True,
        text=True,
        cwd=proteina_dir,
    )
    exit_code = result.returncode
    
    if exit_code != 0:
        error_msg = f"ProteinMPNN failed with exit code {exit_code}\n"
        error_msg += f"Command: {' '.join(cmd_parts)}\n"
        error_msg += f"Working directory: {proteina_dir}\n"
        error_msg += f"PDB file: {pdb_file_path} (exists: {os.path.exists(pdb_file_path)})\n"
        if result.stdout:
            error_msg += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            error_msg += f"STDERR:\n{result.stderr}\n"
        raise RuntimeError(error_msg)
    
    # Wait a bit and check if file exists
    output_file = os.path.join(out_dir_root, "seqs", name + ".fa")
    max_wait = 60  # Wait up to 60 seconds
    wait_time = 0
    while not os.path.exists(output_file) and wait_time < max_wait:
        time.sleep(0.5)
        wait_time += 0.5
    
    if not os.path.exists(output_file):
        # List files in output directory for debugging
        seqs_dir = os.path.join(out_dir_root, "seqs")
        existing_files = []
        if os.path.exists(seqs_dir):
            existing_files = os.listdir(seqs_dir)
        
        error_msg = f"ProteinMPNN output file not found after {max_wait}s: {output_file}\n"
        error_msg += f"Command: {' '.join(cmd_parts)}\n"
        error_msg += f"Working directory: {proteina_dir}\n"
        error_msg += f"Output directory (absolute): {out_dir_root}\n"
        error_msg += f"Output directory exists: {os.path.exists(out_dir_root)}\n"
        error_msg += f"Seqs directory exists: {os.path.exists(seqs_dir)}\n"
        if existing_files:
            error_msg += f"Files in seqs directory: {existing_files[:10]}\n"  # Show first 10 files
        else:
            error_msg += f"Seqs directory is empty or does not exist.\n"
        if result.stdout:
            error_msg += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            error_msg += f"STDERR:\n{result.stderr}\n"
        raise FileNotFoundError(error_msg)
    
    return extract_gen_seqs(output_file)


## ## ## ## ## ## ## ## ## ## ## ##


# ESMFold
## ## ## ## ## ## ## ## ## ## ## ##


# I got this function from hugging face's ESM notebook example
def convert_outputs_to_pdb(outputs) -> List[str]:
    """Takes ESMFold outputs and converts them to a list of PDBs (as strings)."""
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


def run_and_store_esm(
    name: str,
    seqs: List[str],
    path_to_esmfold_out: str,
) -> List[str]:
    """
    Runs ESMFold and stores results as PDB files.

    For now, runs with a single GPU, though not a big deal if we parallelie jobs (easily
    done with our inference pipeline).

    Args:
        name: name to use when storing
        seqs: List of sequences (strings)
        path_to_esmfold_out: Root directory to store outputs of ESMFold as PDBs

    Returns:
        List of paths (list of str) to PDB files
    """
    is_cluster_run = os.environ.get("SLURM_JOB_ID") is not None
    cache_dir = None
    if is_cluster_run:
        cache_dir = os.environ.get("CACHE_DIR")
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/esmfold_v1", cache_dir=cache_dir
    )
    esm_model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1", cache_dir=cache_dir
    )
    esm_model = esm_model.cuda()

    # Run ESMFold
    len(seqs)
    max_nres = max([len(x) for x in seqs]) if seqs else 0
    list_of_strings_pdb = []
    
    # Determine batch size based on sequence length
    if max_nres > 700:
        batch_size = 1
    elif max_nres > 500:
        batch_size = 2
    else:
        batch_size = 4
    
    # Process all sequences in batches
    total_seqs = len(seqs)
    num_batches = (total_seqs + batch_size - 1) // batch_size  # Ceiling division
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, total_seqs)  # Don't go past end
        
        if start_idx >= total_seqs:
            break  # Safety check
        
        batch_seqs = seqs[start_idx:end_idx]
        if not batch_seqs:
            break
        
        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )
        inputs = {k: inputs[k].cuda() for k in inputs}

        with torch.no_grad():
            _outputs = esm_model(**inputs)

        _list_of_strings_pdb = convert_outputs_to_pdb(_outputs)
        list_of_strings_pdb.extend(_list_of_strings_pdb)

    # Create out directory if not there
    if not os.path.exists(path_to_esmfold_out):
        os.makedirs(path_to_esmfold_out)

    # Store generations for each sequence
    out_esm_paths = []
    for i, pdb in enumerate(list_of_strings_pdb):
        fname = f"esm_{i+1}.pdb_esm"
        fdir = os.path.join(path_to_esmfold_out, fname)
        with open(fdir, "w") as f:
            f.write(pdb)
            out_esm_paths.append(fdir)
    return out_esm_paths


## ## ## ## ## ## ## ## ## ## ## ##


def load_pdb(fname: str) -> str:
    """Returns pdb stored in input file as string."""
    with open(fname, "r") as f:
        return from_pdb_string(f.read())


def rmsd_metric(
    coors_1_atom37: Float[Tensor, "n 37 3"],
    coors_2_atom37: Float[Tensor, "n 37 3"],
    mask_1_atom_37: Optional[Float[Tensor, "n 37"]] = None,
    mask_2_atom_37: Optional[Float[Tensor, "n 37"]] = None,
    mode: str = "ca",
    incl_ox: bool = False,
    align: bool = True,
) -> Float[Tensor, ""]:
    """
    Computes RMSD between two protein structures in the Atom37 represnetation.
    For now we only use mask to check whether we have all required atoms.

    Args:
        coors_1_atom37: First structure, shape [n, 37, 3]
        coors_2_atom37: Second structure, shape [n, 37, 3]
        mask_1_atom37: Binary mask of first structure, shape [n, 37]
        mask_2_atom37: Binary mask of first structure, shape [n, 37]
        mode: Modality to use, for now single option "ca", referring to only alpha
            carbon.
        incl_ox: Wehther to include oxygen atom, should be left to False
        align: Whether to align pointclouds before computing RMSD.

    Returns:
        RMSD value, as a Torch (float) tensor with a single element
    """
    assert coors_1_atom37.shape == coors_2_atom37.shape
    assert coors_1_atom37.shape[-1] == 3
    assert coors_1_atom37.shape[-2] == 37
    if mask_1_atom_37 is not None:
        assert mask_1_atom_37.shape == coors_1_atom37.shape[1:]
    if mask_2_atom_37 is not None:
        assert mask_2_atom_37.shape == coors_2_atom37.shape[1:]
    idx_select = [1]  # [CA]

    coors_1 = coors_1_atom37[:, idx_select, :]  # [n, natoms_sel, 3]
    coors_2 = coors_2_atom37[:, idx_select, :]  # [n, natoms_sel, 3]

    # Check all atoms actually present if we have mask
    for mask_atom_37 in [mask_1_atom_37, mask_2_atom_37]:
        if mask_atom_37 is not None:
            mask = mask_atom_37[:, idx_select]
            assert mask.sum() == mask.numel()

    # Compute RMSD (potentially) aligning structures
    coors_1 = einops.rearrange(coors_1, "n s t -> (n s) t")  # [n * natoms_sel, 3]
    coors_2 = einops.rearrange(coors_2, "n s t -> (n s) t")  # [n * natoms_sel, 3]

    if align:
        coors_1, coors_2 = kabsch_align_ind(coors_1, coors_2, ret_both=True)

    sq_err = (coors_1 - coors_2) ** 2
    return sq_err.sum(dim=-1).mean().sqrt().item()


def scRMSD(
    pdb_file_path: str,
    tmp_path: str = "./tmp/metrics/",
    num_seq_per_target: int = 8,
    pmpnn_sampling_temp: float = 0.1,
    ret_min=True,
    path_to_model_weights: Optional[str] = None,
) -> Union[float, List[float]]:
    """
    Evaluates self-consistency RMSD metrics for given pdb.

    Args:
        pdb_file_path: Path to PDB file.
        tmp_path: Path to store files produced by ProteinMPNN and ESMFold.
        num_seq_per_target: Number of sequences generated by ProteinMPNN per structure.
        pmpnn_sampling_temp: ProteinMPNN sampling temperature.
        ret_min: Whether to return min RMSD or a list of all values.
        path_to_model_weights: Optional path to ProteinMPNN model weights directory.
            If not provided, ProteinMPNN will look for weights in its default location.

    Returns:
        Either best RMSD (scRMSD) or a list of all values for all generations, depending on
        the ret_min argument.
    """
    name = pdb_name_from_path(pdb_file_path)
    scrmd_start = time.time()

    mpnn_start = time.time()
    logger.info("Running ProteinMPNN")
    mpnn_gen_seqs = run_proteinmpnn(  # For now do not use keep ca_only=False
        pdb_file_path,
        tmp_path,
        num_seq_per_target=num_seq_per_target,
        sampling_temp=pmpnn_sampling_temp,
        path_to_model_weights=path_to_model_weights,
    )  # List of sequences
    mpnn_time = time.time() - mpnn_start

    esm_start = time.time()
    logger.info(f"Running ESMFold for {name}")
    out_esm_paths = run_and_store_esm(name, mpnn_gen_seqs, tmp_path)
    # List of paths to PDBs
    esm_time = time.time() - esm_start

    # Compute RMSDs
    rmsd_start = time.time()
    results = []

    # Load generated
    gen_prot = load_pdb(pdb_file_path)
    gen_coors = torch.Tensor(gen_prot.atom_positions)

    # Load ESMs
    for out_esm in out_esm_paths:
        try:
            rec_prot_esm = load_pdb(out_esm)
            rec_coors = torch.Tensor(rec_prot_esm.atom_positions)

            # Validate coordinate shapes before calling rmsd_metric
            # This can fail when ESMFold produces malformed structures (e.g., after many runs
            # when GPU memory gets fragmented or resources are exhausted)
            if gen_coors.shape != rec_coors.shape:
                logger.warning(
                    f"Shape mismatch for {name}: gen_coors.shape={gen_coors.shape}, "
                    f"rec_coors.shape={rec_coors.shape} (ESMFold output: {out_esm}). "
                    f"Skipping this comparison."
                )
                continue
            
            # Additional validation: check that coordinates have expected dimensions
            if len(gen_coors.shape) != 3 or gen_coors.shape[-1] != 3 or gen_coors.shape[-2] != 37:
                logger.warning(
                    f"Invalid gen_coors shape for {name}: {gen_coors.shape}. Expected [n, 37, 3]. Skipping."
                )
                continue
            
            if len(rec_coors.shape) != 3 or rec_coors.shape[-1] != 3 or rec_coors.shape[-2] != 37:
                logger.warning(
                    f"Invalid rec_coors shape for {name} (ESMFold output: {out_esm}): {rec_coors.shape}. "
                    f"Expected [n, 37, 3]. Skipping."
                )
                continue
            
            results.append(rmsd_metric(gen_coors, rec_coors))  # rmsd_ca
        except (AssertionError, ValueError, RuntimeError, Exception) as e:
            logger.warning(
                f"RMSD computation failed for {name} (ESMFold output: {out_esm}): {type(e).__name__}: {e}. "
                f"Shapes: gen_coors={gen_coors.shape if 'gen_coors' in locals() else 'N/A'}, "
                f"rec_coors={rec_coors.shape if 'rec_coors' in locals() else 'N/A'}. Skipping."
            )
            continue
    rmsd_time = time.time() - rmsd_start
    
    total_time = time.time() - scrmd_start
    
    best_rmsd = min(results) if results else float('inf')
    reward = float(torch.exp(-torch.tensor(best_rmsd))) if torch.isfinite(torch.tensor(best_rmsd)) else 0.0
    
    logger.info(f"[scRMSD] Timing for {name}:")
    logger.info(f"  ProteinMPNN: {mpnn_time:.2f}s ({mpnn_time/total_time*100:.1f}%)")
    logger.info(f"  ESMFold: {esm_time:.2f}s ({esm_time/total_time*100:.1f}%)")
    logger.info(f"  RMSD computation: {rmsd_time:.2f}s ({rmsd_time/total_time*100:.1f}%)")
    logger.info(f"  Total: {total_time:.2f}s | Best RMSD: {best_rmsd:.4f} | Reward: {reward:.4f}")

    if ret_min:
        return min(results)
    return results


def scRMSD_batch(
    pdb_file_paths: List[str],
    tmp_path: str = "./tmp/metrics/",
    num_seq_per_target: int = 8,
    pmpnn_sampling_temp: float = 0.1,
    path_to_model_weights: Optional[str] = None,
) -> List[float]:
    """
    Batch version of scRMSD that processes multiple PDB files efficiently.
    
    This function batches ESMFold calls across all sequences from all proteins,
    which is much faster than calling scRMSD individually for each protein.

    Args:
        pdb_file_paths: List of paths to PDB files.
        tmp_path: Path to store files produced by ProteinMPNN and ESMFold.
        num_seq_per_target: Number of sequences generated by ProteinMPNN per structure.
        pmpnn_sampling_temp: ProteinMPNN sampling temperature.
        path_to_model_weights: Optional path to ProteinMPNN model weights directory.

    Returns:
        List of best RMSD scores (one per input PDB file), converted to rewards via exp(-rmsd).
    """
    if not pdb_file_paths:
        return []
    
    batch_start = time.time()
    
    # Step 1: Run ProteinMPNN for all PDBs in parallel
    mpnn_start = time.time()
    logger.info(f"Running ProteinMPNN for {len(pdb_file_paths)} proteins in parallel")
    
    def run_mpnn_single(pdb_file_path: str) -> tuple[str, List[str], float]:
        """Run ProteinMPNN for a single PDB and return (name, sequences, time)."""
        name = pdb_name_from_path(pdb_file_path)
        single_start = time.time()
        try:
            mpnn_gen_seqs = run_proteinmpnn(
                pdb_file_path,
                tmp_path,
                num_seq_per_target=num_seq_per_target,
                sampling_temp=pmpnn_sampling_temp,
                path_to_model_weights=path_to_model_weights,
            )
            single_time = time.time() - single_start
            return (name, mpnn_gen_seqs, single_time)
        except Exception as e:
            logger.error(f"ProteinMPNN failed for {name}: {e}")
            raise
    
    # Run ProteinMPNN calls in parallel (using ThreadPoolExecutor since subprocess calls are I/O bound)
    # Note: ProteinMPNN uses GPU; 8 workers balances throughput vs GPU memory on typical nodes
    max_workers = min(len(pdb_file_paths), 8)
    logger.info(f"  Using {max_workers} parallel workers for ProteinMPNN")
    
    all_mpnn_seqs = []  # List of (pdb_name, sequences_list) tuples
    all_names = []
    results_dict = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_pdb = {
            executor.submit(run_mpnn_single, pdb_path): pdb_path 
            for pdb_path in pdb_file_paths
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_pdb):
            pdb_path = future_to_pdb[future]
            completed += 1
            try:
                name, seqs, single_time = future.result()
                results_dict[pdb_path] = (name, seqs, single_time)
                logger.info(f"  ProteinMPNN {completed}/{len(pdb_file_paths)} ({name}): {single_time:.2f}s")
            except Exception as e:
                logger.error(f"ProteinMPNN failed for {pdb_path}: {e}")
                raise
    
    # Reconstruct in original order
    for pdb_file_path in pdb_file_paths:
        name, seqs, _ = results_dict[pdb_file_path]
        all_names.append(name)
        all_mpnn_seqs.append((name, seqs))
    
    mpnn_total_time = time.time() - mpnn_start
    
    # Step 2: Collect all sequences with their corresponding protein names
    # Format: (protein_name, sequence_index, sequence)
    all_seqs_with_metadata = []
    for name, seqs in all_mpnn_seqs:
        for seq_idx, seq in enumerate(seqs):
            all_seqs_with_metadata.append((name, seq_idx, seq))
    
    if not all_seqs_with_metadata:
        # No sequences generated, return worst scores
        return [0.0] * len(pdb_file_paths)
    
    # Step 3: Batch run ESMFold for all sequences at once
    esm_start = time.time()
    logger.info(f"Running ESMFold for {len(all_seqs_with_metadata)} sequences from {len(pdb_file_paths)} proteins")
    all_seqs = [seq for _, _, seq in all_seqs_with_metadata]
    
    # Use a single name prefix for batch ESMFold, but we'll track which sequences belong to which protein
    # The name parameter isn't actually used in file naming, but we pass it for consistency
    # Note: run_and_store_esm creates files directly in tmp_path, same as original scRMSD
    batch_name = f"batch_{len(pdb_file_paths)}proteins"
    out_esm_paths = run_and_store_esm(batch_name, all_seqs, tmp_path)
    esm_time = time.time() - esm_start
    
    # Step 4: Group ESMFold outputs back by protein and compute RMSDs
    # Map each protein to its ESMFold outputs
    protein_esm_paths = {name: [] for name in all_names}
    for (name, seq_idx, _), esm_path in zip(all_seqs_with_metadata, out_esm_paths):
        protein_esm_paths[name].append(esm_path)
    
    # Step 5: Compute RMSD for each protein
    rmsd_start = time.time()
    all_results = []
    for pdb_file_path, name in zip(pdb_file_paths, all_names):
        gen_prot = load_pdb(pdb_file_path)
        gen_coors = torch.Tensor(gen_prot.atom_positions)
        
        # Compute RMSD for all ESMFold outputs for this protein
        protein_results = []
        num_esm_outputs = len(protein_esm_paths[name])
        if num_esm_outputs == 0:
            logger.warning(f"No ESMFold outputs for {name}, this indicates a bug in batch processing")
            all_results.append(0.0)
            continue
            
        for out_esm in protein_esm_paths[name]:
            try:
                rec_prot_esm = load_pdb(out_esm)
                rec_coors = torch.Tensor(rec_prot_esm.atom_positions)
                
                # Validate coordinate shapes before calling rmsd_metric
                # This can fail when ESMFold produces malformed structures (e.g., after many runs
                # when GPU memory gets fragmented or resources are exhausted)
                if gen_coors.shape != rec_coors.shape:
                    logger.warning(
                        f"Shape mismatch for {name}: gen_coors.shape={gen_coors.shape}, "
                        f"rec_coors.shape={rec_coors.shape} (ESMFold output: {out_esm}). "
                        f"Skipping this comparison."
                    )
                    continue
                
                # Additional validation: check that coordinates have expected dimensions
                if len(gen_coors.shape) != 3 or gen_coors.shape[-1] != 3 or gen_coors.shape[-2] != 37:
                    logger.warning(
                        f"Invalid gen_coors shape for {name}: {gen_coors.shape}. Expected [n, 37, 3]. Skipping."
                    )
                    continue
                
                if len(rec_coors.shape) != 3 or rec_coors.shape[-1] != 3 or rec_coors.shape[-2] != 37:
                    logger.warning(
                        f"Invalid rec_coors shape for {name} (ESMFold output: {out_esm}): {rec_coors.shape}. "
                        f"Expected [n, 37, 3]. Skipping."
                    )
                    continue
                
                protein_results.append(rmsd_metric(gen_coors, rec_coors))
            except (AssertionError, ValueError, RuntimeError, Exception) as e:
                # gen_coors is always available (defined in outer loop)
                # rec_coors may not be available if load_pdb failed
                try:
                    rec_shape_str = str(rec_coors.shape)
                except NameError:
                    rec_shape_str = 'N/A (failed to load)'
                logger.warning(
                    f"RMSD computation failed for {name} (ESMFold output: {out_esm}): "
                    f"{type(e).__name__}: {e}. Shapes: gen_coors={gen_coors.shape}, "
                    f"rec_coors={rec_shape_str}. Skipping."
                )
                continue
        
        if protein_results:
            best_rmsd = min(protein_results)
            # Convert to reward: exp(-rmsd)
            reward = float(torch.exp(-torch.tensor(best_rmsd)))
            all_results.append(reward)
            logger.debug(f"  {name}: {num_esm_outputs} ESMFold outputs, best RMSD: {best_rmsd:.4f}, reward: {reward:.4f}")
        else:
            # No valid results, return worst score
            logger.warning(f"No valid RMSD results for {name}")
            all_results.append(0.0)
    rmsd_time = time.time() - rmsd_start
    
    total_time = time.time() - batch_start
    
    # Compute designability percentage (as in paper: % with scRMSD < threshold)
    # Common thresholds: 2.0 Å (strict), 3.0 Å (moderate)
    rmsd_values = [-np.log(r) if r > 0 else float('inf') for r in all_results]  # Convert rewards back to RMSD
    designability_2A = sum(1 for rmsd in rmsd_values if rmsd < 2.0) / len(rmsd_values) * 100
    designability_3A = sum(1 for rmsd in rmsd_values if rmsd < 3.0) / len(rmsd_values) * 100
    
    # Print detailed timing breakdown
    logger.info(f"[scRMSD_batch] Timing breakdown for {len(pdb_file_paths)} proteins:")
    logger.info(f"  ProteinMPNN: {mpnn_total_time:.2f}s ({mpnn_total_time/total_time*100:.1f}%)")
    logger.info(f"  ESMFold: {esm_time:.2f}s ({esm_time/total_time*100:.1f}%)")
    logger.info(f"  RMSD computation: {rmsd_time:.2f}s ({rmsd_time/total_time*100:.1f}%)")
    logger.info(f"  Total: {total_time:.2f}s")
    logger.info(f"  Rewards: {[f'{r:.4f}' for r in all_results]}")
    logger.info(f"  Designability: {designability_2A:.1f}% (<2Å), {designability_3A:.1f}% (<3Å)")
    
    return all_results
