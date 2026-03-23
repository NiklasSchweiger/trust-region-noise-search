#!/bin/bash
# ==============================================================================
# Diversity and Novelty Computation for Protein Experiments
# ==============================================================================
# Computes two metrics on a directory of generated protein structures (.pdb):
#   Diversity  – cluster count / total structures (TM-score clustering via foldseek)
#              – mean pairwise TM-score
#   Novelty    – mean max TM-score against the PDB (lower = more novel)
#
# Reports both unfiltered results and results restricted to designable structures
# (designability > 0.1353, equivalent to scRMSD < 2 Å, matching Proteina's methodology).
#
# Requirements:
#   - foldseek  (https://github.com/steineggerlab/foldseek)
#   - A local copy of the PDB foldseek database for novelty computation.
#     Download: foldseek databases PDB /path/to/pdb_db /tmp/foldseek_tmp
#     Then set: export FOLDSEEK_DB=/path/to/pdb_db
#
# Usage (run from repo root):
#   export FOLDSEEK_DB=/path/to/foldseek/pdb_db
#   ./scripts/compute_diversity_novelty.sh <output_directory> [job_id]
#
# The output_directory should contain *best*_proteina.pdb files produced by
# scripts/run_protein_benchmarks.sh.  Results are written to:
#   outputs/diversity_results/[job_<job_id>/]fixed_<length>/<solver>_<reward>_<mode>/
# ==============================================================================

set -e

# Run from repo root so outputs/ and tmp/ paths are correct
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ==============================================================================
# CONFIGURABLE PARAMETERS
# ==============================================================================

FOLDSEEK_DB="${FOLDSEEK_DB:-/path/to/foldseek/pdb_db}"
# Results are saved separately from PDB files to avoid confusion
# Structure: outputs/diversity_results_streamlined/[job_<job_id>/]fixed_<length>/<solver>_<reward>_<mode>/
RESULTS_BASE="outputs/diversity_results"
TMP_BASE="tmp/foldseek_tmp"

# Clustering parameters
TM_SCORE_THRESHOLD=0.5
ALIGNMENT_TYPE=1
COV_MODE=0
MIN_SEQ_ID=0

# Designability filtering threshold (matches Proteina: scRMSD < 2Å → designability > exp(-2) ≈ 0.1353)
DESIGNABILITY_THRESHOLD=0.1353

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Extract designability score from PDB filename
# Format: *rew<VALUE>* (e.g., 50res_best_rew0.6755_proteina.pdb → 0.6755)
# Note: designability = exp(-scrmsd), so we extract the reward value directly
extract_designability() {
    local filename="$1"
    # Try to match pattern: rew<VALUE> where VALUE is a decimal number
    if [[ "$filename" =~ rew([0-9]+\.[0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    elif [[ "$filename" =~ rew([0-9]+) ]]; then
        # If integer only, convert to float
        echo "${BASH_REMATCH[1]}.0"
    else
        # If not found, return 0.0 (will be filtered out if threshold > 0)
        echo "0.0"
    fi
}

# ==============================================================================
# CHECK DEPENDENCIES
# ==============================================================================

# Activate proteina environment to access foldseek (skip if already active)
if command -v micromamba &> /dev/null && [[ -z "$CONDA_DEFAULT_ENV" || "$CONDA_DEFAULT_ENV" != "proteina_env" ]]; then
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate proteina_env 2>/dev/null || true
elif command -v conda &> /dev/null; then
    conda activate proteina_env 2>/dev/null || true
fi

# Check for foldseek
FOLDSEEK_CMD="foldseek"
if ! command -v "$FOLDSEEK_CMD" &> /dev/null; then
    log_error "foldseek not found. Please activate proteina_env: micromamba activate proteina_env"
    exit 1
fi

# ==============================================================================
# PARSE ARGUMENTS
# ==============================================================================

if [[ $# -lt 1 ]]; then
    log_error "Usage: $0 <source_directory> [job_id]"
    echo ""
    echo "Example (from repo root):"
    echo "  ./scripts/compute_diversity_novelty.sh outputs/proteins/prot_fix50_designability_trs_sc"
    echo "  ./scripts/compute_diversity_novelty.sh outputs/proteins/prot_fix50_designability_trs_sc_job12345 12345"
    echo ""
    echo "Results are saved to: outputs/diversity_results_streamlined/[job_<job_id>/]fixed_<length>/..."
    exit 1
fi

SOURCE_DIR="$1"
JOB_ID_ARG="$2"

# Use provided job ID or try to extract from SLURM/environment
if [[ -n "$JOB_ID_ARG" ]]; then
    JOB_ID="$JOB_ID_ARG"
elif [[ -n "$SLURM_JOB_ID" ]]; then
    JOB_ID="$SLURM_JOB_ID"
elif [[ -n "$JOB_ID" ]]; then
    # Already set from environment
    :
else
    # Try to extract job ID from source directory name (format: ..._job<ID>)
    if [[ "$SOURCE_DIR" =~ _job([0-9]+) ]]; then
        JOB_ID="${BASH_REMATCH[1]}"
    fi
fi

# Update results base if job ID is available
if [[ -n "$JOB_ID" ]]; then
    RESULTS_BASE="${RESULTS_BASE}/job_${JOB_ID}"
    log_info "Using job ID: $JOB_ID"
    log_info "Results will be saved to: $RESULTS_BASE"
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
    log_error "Source directory not found: $SOURCE_DIR"
    exit 1
fi

# ==============================================================================
# PARSE DIRECTORY NAME
# ==============================================================================

# Extract information from directory name
# Format: prot_fix50_designability_random_search_setting1_sc
RUN_NAME=$(basename "$SOURCE_DIR")

FIXED_LENGTH=""
SOLVER=""
REWARD=""
SAMPLING_MODE="unknown"

# Check for _sc or _vf suffix
if [[ "$RUN_NAME" =~ _sc(_job[0-9]+)?$ ]]; then
    SAMPLING_MODE="sc"
    BASE_NAME=$(echo "$RUN_NAME" | sed -E 's/_sc(_job[0-9]+)?$//')
elif [[ "$RUN_NAME" =~ _vf(_job[0-9]+)?$ ]]; then
    SAMPLING_MODE="vf"
    BASE_NAME=$(echo "$RUN_NAME" | sed -E 's/_vf(_job[0-9]+)?$//')
elif [[ "$RUN_NAME" =~ _job([0-9]+)$ ]]; then
    # Try to find sampling mode before _job
    if [[ "$RUN_NAME" == *_sc_job* ]]; then
        SAMPLING_MODE="sc"
        BASE_NAME=$(echo "$RUN_NAME" | sed -E 's/_sc_job[0-9]+$//')
    elif [[ "$RUN_NAME" == *_vf_job* ]]; then
        SAMPLING_MODE="vf"
        BASE_NAME=$(echo "$RUN_NAME" | sed -E 's/_vf_job[0-9]+$//')
    else
        BASE_NAME=$(echo "$RUN_NAME" | sed -E 's/_job[0-9]+$//')
    fi
else
    BASE_NAME="$RUN_NAME"
fi

# Parse: prot_fix50_designability_random_search_setting1
if [[ "$BASE_NAME" =~ prot_fix([0-9]+)_([^_]+)_(.+)_setting ]]; then
    FIXED_LENGTH="${BASH_REMATCH[1]}"
    REWARD="${BASH_REMATCH[2]}"
    SOLVER="${BASH_REMATCH[3]}"
else
    log_warning "Could not parse directory name, using defaults"
    FIXED_LENGTH="unknown"
    SOLVER="unknown"
    REWARD="unknown"
fi

log_info "Processing: $RUN_NAME"
log_info "  Fixed length: $FIXED_LENGTH"
log_info "  Solver: $SOLVER"
log_info "  Reward: $REWARD"
log_info "  Sampling mode: $SAMPLING_MODE"

# ==============================================================================
# FIND VALID PDB FILES
# ==============================================================================

# Only process files with both "best" and "_proteina" in the name (valid best structures)
PDB_FILES=()
while IFS= read -r -d '' file; do
    if [[ "$(basename "$file")" == *"best"* ]]; then
        PDB_FILES+=("$file")
    fi
done < <(find "$SOURCE_DIR" -name "*_proteina.pdb" -type f -print0)

NUM_PDBS=${#PDB_FILES[@]}

if [[ $NUM_PDBS -eq 0 ]]; then
    log_error "No *best*_proteina.pdb files found in $SOURCE_DIR"
    exit 1
fi

log_success "Found $NUM_PDBS valid PDB files (*best*_proteina.pdb)"

# Filter files by designability threshold (matches Proteina methodology)
log_info "Filtering files by designability threshold (> $DESIGNABILITY_THRESHOLD, equivalent to scRMSD < 2Å)..."
PDB_FILES_ALL=("${PDB_FILES[@]}")  # Keep all files
PDB_FILES_DESIGNABLE=()  # Filtered to designable only
DESIGNABLE_COUNT=0
EXTRACTION_FAILED=0

for pdb in "${PDB_FILES[@]}"; do
    filename=$(basename "$pdb")
    designability=$(extract_designability "$filename")
    
    # Check if extraction failed (returned 0.0 but might be missing from filename)
    if [[ "$designability" == "0.0" ]]; then
        # Check if filename actually contains "rew" pattern
        if [[ ! "$filename" =~ rew ]]; then
            EXTRACTION_FAILED=$((EXTRACTION_FAILED + 1))
        fi
    fi
    
    # Compare designability (handle floating point comparison)
    if (( $(echo "$designability > $DESIGNABILITY_THRESHOLD" | bc -l) )); then
        PDB_FILES_DESIGNABLE+=("$pdb")
        DESIGNABLE_COUNT=$((DESIGNABLE_COUNT + 1))
    fi
done

log_info "  All structures: $NUM_PDBS"
log_info "  Designable structures (designability > $DESIGNABILITY_THRESHOLD): $DESIGNABLE_COUNT"
log_info "  Non-designable structures: $((NUM_PDBS - DESIGNABLE_COUNT))"
if [[ $EXTRACTION_FAILED -gt 0 ]]; then
    log_warning "  Warning: Could not extract designability from $EXTRACTION_FAILED filenames (assuming non-designable)"
fi

# Function to compute diversity and novelty for a given set of PDB files
# Arguments: 
#   $1: array name of PDB files (use array name, not array itself)
#   $2: number of files
#   $3: suffix for output directories (e.g., "_all" or "_designable")
#   $4: description string for logging
compute_diversity_novelty() {
    local files_array_name="$1[@]"
    local num_files="$2"
    local suffix="$3"
    local description="$4"
    
    local -a input_files=("${!files_array_name}")
    
    if [[ $num_files -eq 0 ]]; then
        log_warning "No files to process for $description, skipping..."
        return 1
    fi
    
    log_info "========================================="
    log_info "Computing metrics for $description ($num_files structures)"
    log_info "========================================="
    
    # Create temporary directory for this computation
    local tmp_pdb_dir="$TMP_BASE/${FIXED_LENGTH}_${SOLVER}_${REWARD}_${SAMPLING_MODE}/pdbs${suffix}"
    rm -rf "$tmp_pdb_dir"
    mkdir -p "$tmp_pdb_dir"
    
    # Copy files to temp directory with UNIQUE names
    local -a tmp_pdb_files=()
    for pdb in "${input_files[@]}"; do
        task_dir="$(basename "$(dirname "$pdb")")"
        base="$(basename "$pdb")"
        tmp_name="${task_dir}__${base}"
        tmp_path="$tmp_pdb_dir/$tmp_name"
        cp "$pdb" "$tmp_path"
        tmp_pdb_files+=("$tmp_path")
    done
    
    # Setup output directories
    local out_dir="$RESULTS_BASE/fixed_${FIXED_LENGTH}/${SOLVER}_${REWARD}_${SAMPLING_MODE}${suffix}"
    local cluster_dir="$out_dir/cluster_diversity"
    local novelty_dir="$out_dir/novelty"
    local pairwise_dir="$out_dir/pairwise_diversity"
    local tmp_dir="$TMP_BASE/${FIXED_LENGTH}_${SOLVER}_${REWARD}_${SAMPLING_MODE}${suffix}"
    
    mkdir -p "$cluster_dir" "$novelty_dir" "$pairwise_dir"
    
    # Initialize return variables (using global namespace)
    eval "NUM_CLUSTERS${suffix}=0"
    eval "DIVERSITY_RATIO${suffix}=\"NA\""
    eval "PAIRWISE_TMSCORE_DIVERSITY${suffix}=\"NA\""
    eval "NOVELTY_AVG${suffix}=\"NA\""
    eval "NOVELTY_MIN${suffix}=\"NA\""
    eval "NOVELTY_MAX${suffix}=\"NA\""
    eval "NOVELTY_MATCHED_ONLY${suffix}=\"NA\""
    eval "PCT_NO_MATCH${suffix}=\"NA\""
    
    # 1. CLUSTER DIVERSITY
    log_info "Computing cluster diversity..."
    rm -f "$cluster_dir"/res*
    rm -rf "$tmp_dir/cluster_tmp"
    mkdir -p "$tmp_dir/cluster_tmp"
    
    $FOLDSEEK_CMD easy-cluster \
        "$tmp_pdb_dir" \
        "$cluster_dir/res" \
        "$tmp_dir/cluster_tmp" \
        --alignment-type $ALIGNMENT_TYPE \
        --cov-mode $COV_MODE \
        --min-seq-id $MIN_SEQ_ID \
        --tmscore-threshold $TM_SCORE_THRESHOLD \
        2>/dev/null
    
    local num_clusters=0
    local diversity_ratio="NA"
    
    if [[ -f "$cluster_dir/res_cluster.tsv" ]]; then
        num_clusters=$(cut -f1 "$cluster_dir/res_cluster.tsv" | sort -u | wc -l)
        local num_members=$(cut -f2 "$cluster_dir/res_cluster.tsv" | sort -u | wc -l)
        
        if [[ $num_members -ne $num_files ]]; then
            log_error "ERROR: Members in cluster file ($num_members) != input structures ($num_files)!"
            return 1
        fi
        
        diversity_ratio=$(echo "scale=4; $num_clusters / $num_files" | bc)
    elif [[ -f "$cluster_dir/res_clu.tsv" ]]; then
        num_clusters=$(cut -f1 "$cluster_dir/res_clu.tsv" | sort -u | wc -l)
        local num_members=$(cut -f2 "$cluster_dir/res_clu.tsv" | sort -u | wc -l)
        
        if [[ $num_members -ne $num_files ]]; then
            log_error "ERROR: Members in cluster file ($num_members) != input structures ($num_files)!"
            return 1
        fi
        
        diversity_ratio=$(echo "scale=4; $num_clusters / $num_files" | bc)
    else
        log_error "Clustering failed - no output file generated"
        return 1
    fi
    
    eval "NUM_CLUSTERS${suffix}=$num_clusters"
    eval "DIVERSITY_RATIO${suffix}=\"$diversity_ratio\""
    log_success "Cluster diversity: $num_clusters clusters / $num_files structures = $diversity_ratio"
    
    # 2. PAIRWISE TM-SCORE DIVERSITY
    if [[ $num_files -lt 2 ]]; then
        log_warning "Need at least 2 structures for pairwise diversity, skipping"
    else
        log_info "Computing pairwise TM-score diversity..."
        
        local pairwise_db="$tmp_dir/pairwise_db"
        $FOLDSEEK_CMD createdb "$tmp_pdb_dir" "$pairwise_db" \
            --chain-name-mode 0 \
            --db-extraction-mode 0 \
            --distance-threshold 8 \
            --coord-store-mode 2 \
            --write-lookup 1 \
            2>/dev/null
        
        local pairwise_output="$pairwise_dir/pairwise_tmscores.txt"
        > "$pairwise_output"
        
        local search_out_all="$tmp_dir/pairwise_search_all.txt"
        $FOLDSEEK_CMD easy-search \
            "$tmp_pdb_dir" \
            "$pairwise_db" \
            "$search_out_all" \
            "$tmp_dir/pairwise_tmp" \
            --alignment-type $ALIGNMENT_TYPE \
            --exhaustive-search \
            --tmscore-threshold 0.0 \
            --max-seqs 10000000000 \
            --format-output query,target,alntmscore,lddt \
            2>/dev/null
        
        if [[ -f "$search_out_all" && -s "$search_out_all" ]]; then
            awk -F'\t' '$1 != $2 {print $3}' "$search_out_all" > "$pairwise_output"
        fi
        
        if [[ -f "$pairwise_output" && -s "$pairwise_output" ]]; then
            local pairwise_tmscore=$(awk '
                BEGIN { sum=0; count=0 }
                { sum += $1; count++ }
                END { if (count > 0) printf "%.4f", sum/count; else print "NA" }
            ' "$pairwise_output")
            
            eval "PAIRWISE_TMSCORE_DIVERSITY${suffix}=\"$pairwise_tmscore\""
            if [[ "$pairwise_tmscore" != "NA" ]]; then
                log_success "Pairwise TM-score diversity: $pairwise_tmscore"
            fi
        fi
    fi
    
    # 3. NOVELTY
    log_info "Computing novelty (TM-score vs PDB)..."
    
    if [[ ! -f "$FOLDSEEK_DB" && ! -f "${FOLDSEEK_DB}.dbtype" && ! -d "$FOLDSEEK_DB" ]]; then
        log_warning "PDB database not found at $FOLDSEEK_DB, skipping novelty computation"
    else
        local -a novelty_scores=()
        local -a matched_scores=()
        local no_match_count=0
        
        local novelty_out_all="$tmp_dir/novelty_search_all.txt"
        $FOLDSEEK_CMD easy-search \
            "$tmp_pdb_dir" \
            "$FOLDSEEK_DB" \
            "$novelty_out_all" \
            "$tmp_dir/novelty_tmp" \
            --alignment-type $ALIGNMENT_TYPE \
            --exhaustive-search \
            --tmscore-threshold 0.0 \
            --max-seqs 10000000000 \
            --format-output query,target,alntmscore,lddt \
            2>/dev/null
        
        if [[ -f "$novelty_out_all" && -s "$novelty_out_all" ]]; then
            awk -F'\t' '
                {
                    if ($3 > max[$1]) max[$1] = $3
                }
                END {
                    for (q in max) print max[q]
                }
            ' "$novelty_out_all" > "$tmp_dir/max_tm_scores.txt"
            
            while read score; do
                if [[ "$score" == "0.0" || "$score" == "0" ]]; then
                    no_match_count=$((no_match_count + 1))
                else
                    matched_scores+=("$score")
                fi
                novelty_scores+=("$score")
            done < "$tmp_dir/max_tm_scores.txt"
            
            local num_matched_queries=$(cut -f1 "$novelty_out_all" | sort -u | wc -l)
            if [[ $num_matched_queries -lt $num_files ]]; then
                local missing=$((num_files - num_matched_queries))
                for ((i=0; i<missing; i++)); do
                    novelty_scores+=("0.0")
                    no_match_count=$((no_match_count + 1))
                done
            fi
        else
            for ((i=0; i<num_files; i++)); do
                novelty_scores+=("0.0")
                no_match_count=$((no_match_count + 1))
            done
        fi
        
        if [[ ${#novelty_scores[@]} -gt 0 ]]; then
            local stats=$(printf '%s\n' "${novelty_scores[@]}" | awk '
                BEGIN { sum=0; min=999; max=0; count=0 }
                {
                    sum += $1
                    count++
                    if ($1 < min) min = $1
                    if ($1 > max) max = $1
                }
                END {
                    if (count > 0) {
                        avg = sum / count
                        printf "%.4f %.4f %.4f", avg, min, max
                    } else {
                        print "NA NA NA"
                    }
                }
            ')
            
            local novelty_avg=$(echo "$stats" | cut -d' ' -f1)
            local novelty_min=$(echo "$stats" | cut -d' ' -f2)
            local novelty_max=$(echo "$stats" | cut -d' ' -f3)
            
            local novelty_matched_only="NA"
            if [[ ${#matched_scores[@]} -gt 0 ]]; then
                novelty_matched_only=$(printf '%s\n' "${matched_scores[@]}" | awk '
                    BEGIN { sum=0; count=0 }
                    { sum += $1; count++ }
                    END {
                        if (count > 0) {
                            printf "%.4f", sum/count
                        } else {
                            print "NA"
                        }
                    }
                ')
            fi
            
            local pct_no_match=$(echo "scale=2; $no_match_count * 100 / ${#novelty_scores[@]}" | bc)
            
            eval "NOVELTY_AVG${suffix}=\"$novelty_avg\""
            eval "NOVELTY_MIN${suffix}=\"$novelty_min\""
            eval "NOVELTY_MAX${suffix}=\"$novelty_max\""
            eval "NOVELTY_MATCHED_ONLY${suffix}=\"$novelty_matched_only\""
            eval "PCT_NO_MATCH${suffix}=\"$pct_no_match\""
            
            log_success "Novelty (avg max TM-score): $novelty_avg (lower = more novel)"
            log_info "  Min: $novelty_min, Max: $novelty_max"
            log_info "  Matched-only avg: $novelty_matched_only"
            log_info "  No PDB match: $no_match_count / ${#novelty_scores[@]} ($pct_no_match%)"
        fi
    fi
    
    return 0
}

# ==============================================================================
# COMPUTE METRICS FOR ALL STRUCTURES
# ==============================================================================

compute_diversity_novelty "PDB_FILES_ALL" "$NUM_PDBS" "_all" "all structures (unfiltered)"
ALL_SUCCESS=$?

# ==============================================================================
# COMPUTE METRICS FOR DESIGNABLE STRUCTURES ONLY
# ==============================================================================

if [[ $DESIGNABLE_COUNT -gt 0 ]]; then
    compute_diversity_novelty "PDB_FILES_DESIGNABLE" "$DESIGNABLE_COUNT" "_designable" "designable structures (designability > $DESIGNABILITY_THRESHOLD)"
    DESIGNABLE_SUCCESS=$?
else
    log_warning "No designable structures found, skipping designable-only metrics"
    DESIGNABLE_SUCCESS=1
fi

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

# Save summary CSV with both filtered and unfiltered metrics
SUMMARY_FILE="$RESULTS_BASE/overall_summary.csv"
if [[ ! -f "$SUMMARY_FILE" ]]; then
    echo "fixed_length,solver,reward,sampling_mode,filter_type,num_structures,num_clusters,diversity_ratio,pairwise_tmscore,novelty_pdb,novelty_pdb_min,novelty_pdb_max,novelty_pdb_matched_only,pct_no_pdb_match" > "$SUMMARY_FILE"
fi

# Extract metrics using eval (variables are set with suffixes in the function)
# Use default "NA" if variables don't exist
if [[ $ALL_SUCCESS -eq 0 ]]; then
    eval "NUM_CLUSTERS_ALL=\${NUM_CLUSTERS_all:-NA}"
    eval "DIVERSITY_RATIO_ALL=\${DIVERSITY_RATIO_all:-NA}"
    eval "PAIRWISE_TMSCORE_ALL=\${PAIRWISE_TMSCORE_DIVERSITY_all:-NA}"
    eval "NOVELTY_AVG_ALL=\${NOVELTY_AVG_all:-NA}"
    eval "NOVELTY_MIN_ALL=\${NOVELTY_MIN_all:-NA}"
    eval "NOVELTY_MAX_ALL=\${NOVELTY_MAX_all:-NA}"
    eval "NOVELTY_MATCHED_ONLY_ALL=\${NOVELTY_MATCHED_ONLY_all:-NA}"
    eval "PCT_NO_MATCH_ALL=\${PCT_NO_MATCH_all:-NA}"
    
    echo "$FIXED_LENGTH,$SOLVER,$REWARD,$SAMPLING_MODE,all,$NUM_PDBS,$NUM_CLUSTERS_ALL,$DIVERSITY_RATIO_ALL,$PAIRWISE_TMSCORE_ALL,$NOVELTY_AVG_ALL,$NOVELTY_MIN_ALL,$NOVELTY_MAX_ALL,$NOVELTY_MATCHED_ONLY_ALL,$PCT_NO_MATCH_ALL" >> "$SUMMARY_FILE"
fi

if [[ $DESIGNABLE_SUCCESS -eq 0 ]]; then
    eval "NUM_CLUSTERS_DESIGNABLE=\${NUM_CLUSTERS_designable:-NA}"
    eval "DIVERSITY_RATIO_DESIGNABLE=\${DIVERSITY_RATIO_designable:-NA}"
    eval "PAIRWISE_TMSCORE_DESIGNABLE=\${PAIRWISE_TMSCORE_DIVERSITY_designable:-NA}"
    eval "NOVELTY_AVG_DESIGNABLE=\${NOVELTY_AVG_designable:-NA}"
    eval "NOVELTY_MIN_DESIGNABLE=\${NOVELTY_MIN_designable:-NA}"
    eval "NOVELTY_MAX_DESIGNABLE=\${NOVELTY_MAX_designable:-NA}"
    eval "NOVELTY_MATCHED_ONLY_DESIGNABLE=\${NOVELTY_MATCHED_ONLY_designable:-NA}"
    eval "PCT_NO_MATCH_DESIGNABLE=\${PCT_NO_MATCH_designable:-NA}"
    
    echo "$FIXED_LENGTH,$SOLVER,$REWARD,$SAMPLING_MODE,designable,$DESIGNABLE_COUNT,$NUM_CLUSTERS_DESIGNABLE,$DIVERSITY_RATIO_DESIGNABLE,$PAIRWISE_TMSCORE_DESIGNABLE,$NOVELTY_AVG_DESIGNABLE,$NOVELTY_MIN_DESIGNABLE,$NOVELTY_MAX_DESIGNABLE,$NOVELTY_MATCHED_ONLY_DESIGNABLE,$PCT_NO_MATCH_DESIGNABLE" >> "$SUMMARY_FILE"
fi

log_success "Results saved to: $SUMMARY_FILE"
log_info "========================================="
log_info "Summary - All Structures (Unfiltered):"
if [[ $ALL_SUCCESS -eq 0 ]]; then
    eval "NUM_CLUSTERS_ALL=\${NUM_CLUSTERS_all:-NA}"
    eval "DIVERSITY_RATIO_ALL=\${DIVERSITY_RATIO_all:-NA}"
    eval "PAIRWISE_TMSCORE_ALL=\${PAIRWISE_TMSCORE_DIVERSITY_all:-NA}"
    eval "NOVELTY_AVG_ALL=\${NOVELTY_AVG_all:-NA}"
    log_info "  Structures: $NUM_PDBS"
    log_info "  Clusters: $NUM_CLUSTERS_ALL (diversity ratio: $DIVERSITY_RATIO_ALL)"
    log_info "  Pairwise TM-score: $PAIRWISE_TMSCORE_ALL"
    log_info "  Novelty (avg max TM): $NOVELTY_AVG_ALL"
else
    log_warning "  Metrics computation failed"
fi
log_info "----------------------------------------"
log_info "Summary - Designable Structures Only (designability > $DESIGNABILITY_THRESHOLD, matches Proteina):"
if [[ $DESIGNABLE_SUCCESS -eq 0 ]]; then
    eval "NUM_CLUSTERS_DESIGNABLE=\${NUM_CLUSTERS_designable:-NA}"
    eval "DIVERSITY_RATIO_DESIGNABLE=\${DIVERSITY_RATIO_designable:-NA}"
    eval "PAIRWISE_TMSCORE_DESIGNABLE=\${PAIRWISE_TMSCORE_DIVERSITY_designable:-NA}"
    eval "NOVELTY_AVG_DESIGNABLE=\${NOVELTY_AVG_designable:-NA}"
    log_info "  Structures: $DESIGNABLE_COUNT"
    log_info "  Clusters: $NUM_CLUSTERS_DESIGNABLE (diversity ratio: $DIVERSITY_RATIO_DESIGNABLE)"
    log_info "  Pairwise TM-score: $PAIRWISE_TMSCORE_DESIGNABLE"
    log_info "  Novelty (avg max TM): $NOVELTY_AVG_DESIGNABLE"
else
    log_warning "  Metrics computation failed or no designable structures"
fi
log_info "========================================="

# Diagnostic: Show cluster size distribution for both filtered and unfiltered
log_info ""
log_info "Cluster Size Distribution (for debugging):"
log_info "----------------------------------------"

# All structures cluster distribution
if [[ $ALL_SUCCESS -eq 0 ]]; then
    eval "NUM_CLUSTERS_ALL=\${NUM_CLUSTERS_all:-NA}"
    if [[ "$NUM_CLUSTERS_ALL" != "NA" && $NUM_CLUSTERS_ALL -gt 0 ]]; then
        OUT_DIR_ALL="$RESULTS_BASE/fixed_${FIXED_LENGTH}/${SOLVER}_${REWARD}_${SAMPLING_MODE}_all"
        CLUSTER_DIR_ALL="$OUT_DIR_ALL/cluster_diversity"
        CLUSTER_FILE_ALL=""
        if [[ -f "$CLUSTER_DIR_ALL/res_cluster.tsv" ]]; then
            CLUSTER_FILE_ALL="$CLUSTER_DIR_ALL/res_cluster.tsv"
        elif [[ -f "$CLUSTER_DIR_ALL/res_clu.tsv" ]]; then
            CLUSTER_FILE_ALL="$CLUSTER_DIR_ALL/res_clu.tsv"
        fi
        
        if [[ -n "$CLUSTER_FILE_ALL" ]]; then
            log_info "All Structures (Unfiltered) - Top 10 clusters:"
            cut -f1 "$CLUSTER_FILE_ALL" | sort | uniq -c | sort -rn | head -10 | while read count rep; do
                log_info "  Cluster '$rep': $count structures"
            done
        fi
    fi
fi

# Designable structures cluster distribution
if [[ $DESIGNABLE_SUCCESS -eq 0 ]]; then
    eval "NUM_CLUSTERS_DESIGNABLE=\${NUM_CLUSTERS_designable:-NA}"
    if [[ "$NUM_CLUSTERS_DESIGNABLE" != "NA" && $NUM_CLUSTERS_DESIGNABLE -gt 0 ]]; then
        OUT_DIR_DESIGNABLE="$RESULTS_BASE/fixed_${FIXED_LENGTH}/${SOLVER}_${REWARD}_${SAMPLING_MODE}_designable"
        CLUSTER_DIR_DESIGNABLE="$OUT_DIR_DESIGNABLE/cluster_diversity"
        CLUSTER_FILE_DESIGNABLE=""
        if [[ -f "$CLUSTER_DIR_DESIGNABLE/res_cluster.tsv" ]]; then
            CLUSTER_FILE_DESIGNABLE="$CLUSTER_DIR_DESIGNABLE/res_cluster.tsv"
        elif [[ -f "$CLUSTER_DIR_DESIGNABLE/res_clu.tsv" ]]; then
            CLUSTER_FILE_DESIGNABLE="$CLUSTER_DIR_DESIGNABLE/res_clu.tsv"
        fi
        
        if [[ -n "$CLUSTER_FILE_DESIGNABLE" ]]; then
            log_info "Designable Structures Only - Top 10 clusters:"
            cut -f1 "$CLUSTER_FILE_DESIGNABLE" | sort | uniq -c | sort -rn | head -10 | while read count rep; do
                log_info "  Cluster '$rep': $count structures"
            done
        fi
    fi
fi

log_info ""
log_info "(Note: If cluster count decreased after adding structures, a new structure likely"
log_info " bridged two previously separate clusters, causing them to merge)"
