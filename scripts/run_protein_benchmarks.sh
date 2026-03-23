#!/bin/bash
# ==============================================================================
# Protein Backbone Design Benchmark (Proteina)
#
# Runs TRS, random_search, zero_order, and a naive sampling baseline across
# fixed-length protein benchmarks (50 and 100 residues) with the designability
# reward.  After all jobs complete, run scripts/compute_diversity_novelty.sh to
# obtain diversity and novelty metrics.
#
# SLURM: Adjust SLURM_PARTITION, SLURM_QOS, and activate_env() below.
# Without SLURM, copy any Python command from a heredoc and run it directly.
#
# Run from repo root: ./scripts/run_protein_benchmarks.sh
# ==============================================================================
set -e

MODALITY=protein
REWARD=designability
BENCHMARKS=("protein_fixed_50" "protein_fixed_100")
SOLVERS=("trs" "random_search" "zero_order")

# Sampling mode: "sc" (SDE, stochastic) or "vf" (ODE, deterministic)
SAMPLING_MODE="sc"
SC_NOISE_SCALE=0.1

# Paper configuration (setting1)
BATCH_SIZE=8
NUM_REGIONS=6
NUM_ITERATIONS=20
WARMUP_BATCHES=4
ORACLE_BUDGET=160
CHECKPOINTS="[40,80,120,160]" # NOTE: TRS can be run with checkpoints, but one needs to adjust the warm-up batches to match the paper results. 

# Fixed-length benchmark: number of independent optimization runs (one per protein)
NUM_SAMPLES=100


# ------------------------------------------------------------------------------
# SLURM configuration -- adjust to your cluster
# ------------------------------------------------------------------------------
SLURM_PARTITION="gpu"
SLURM_QOS=""
SLURM_TIME="2-00:00:00"

activate_env() {
    # Proteina requires its own environment
    micromamba activate proteina_env
}

# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
mkdir -p "${REPO_ROOT}/batch_logs/PROTEIN"

SLURM_QOS_LINE=""
[[ -n "$SLURM_QOS" ]] && SLURM_QOS_LINE="#SBATCH --qos=$SLURM_QOS"
WANDB_PROJECT_BASE="PROTEIN_BENCHMARK"

SAMPLING_SUFFIX="_${SAMPLING_MODE}"
SAMPLING_ARGS="proteina.sampling_mode=$SAMPLING_MODE"
[[ "$SAMPLING_MODE" == "sc" ]] && SAMPLING_ARGS="$SAMPLING_ARGS proteina.sc_scale_noise=$SC_NOISE_SCALE"

echo -e "${BLUE}=== Protein Benchmarks ===${NC}"
echo "Benchmarks: ${BENCHMARKS[*]}  Solvers: ${SOLVERS[*]}"
echo "Sampling: $SAMPLING_MODE  Reward: $REWARD"
echo ""

JOB_COUNT=0

for BENCHMARK in "${BENCHMARKS[@]}"; do
    if [[ "$BENCHMARK" == "protein_fixed_50" ]]; then
        FIXED_LENGTH=50
        WANDB_PROJECT="${WANDB_PROJECT_BASE}_FIXED50"
    else
        FIXED_LENGTH=100
        WANDB_PROJECT="${WANDB_PROJECT_BASE}_FIXED100"
    fi

    for SOLVER in "${SOLVERS[@]}"; do
        JOB_COUNT=$((JOB_COUNT + 1))
        job_name="prot_fix${FIXED_LENGTH}_${REWARD}_${SOLVER}${SAMPLING_SUFFIX}"


        SOLVER_ARGS="solver=$SOLVER solver.batch_size=$BATCH_SIZE solver.num_iterations=$NUM_ITERATIONS"
        if [[ "$SOLVER" == "trs" ]]; then
            SOLVER_ARGS="$SOLVER_ARGS solver.num_regions=$NUM_REGIONS solver.warmup_batches=$WARMUP_BATCHES 
        fi
        BENCHMARK_ARGS="benchmark=protein_fixed_length benchmark.min_length=$FIXED_LENGTH benchmark.max_length=$FIXED_LENGTH benchmark.num_samples=$NUM_SAMPLES num_runs=$NUM_SAMPLES"
        ORACLE_BUDGET_JOB=$ORACLE_BUDGET

        echo -e "${YELLOW}[$JOB_COUNT] $job_name${NC}"

        sbatch --job-name="$job_name" \
               --output="${REPO_ROOT}/batch_logs/PROTEIN/${job_name}.%j.out" \
               --error="${REPO_ROOT}/batch_logs/PROTEIN/${job_name}.%j.err" <<HEREDOC
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=$SLURM_PARTITION
$SLURM_QOS_LINE
#SBATCH --gres=gpu:1
#SBATCH --time=$SLURM_TIME

cd "$REPO_ROOT"
find noise_optimization -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

$(declare -f activate_env)
activate_env

JOB_ID=\${SLURM_JOB_ID:-"local"}
RUN_NAME="${job_name}_job\${JOB_ID}"
echo "Starting: \$RUN_NAME"

srun --unbuffered python -m noise_optimization.main \\
    modality=$MODALITY \\
    reward_function=$REWARD \\
    $BENCHMARK_ARGS \\
    $SOLVER_ARGS \\
    $SAMPLING_ARGS \\
    oracle_budget=$ORACLE_BUDGET_JOB \\
    wandb_project=$WANDB_PROJECT \\
    wandb_name=\${RUN_NAME} \\
    wandb=true \\
    logging.wandb.scaling.enabled=true \\
    logging.wandb.scaling.budget_checkpoints=$CHECKPOINTS \\
    +save_best_structures=true

echo "Completed: \$RUN_NAME"
HEREDOC

        echo -e "${GREEN}Submitted: $job_name${NC}"
    done
    echo ""
done

echo -e "${BLUE}Submitted $JOB_COUNT jobs.  Logs: ${REPO_ROOT}/batch_logs/PROTEIN/${NC}"
echo ""
echo "After all jobs complete, compute diversity and novelty:"
echo "  export FOLDSEEK_DB=/path/to/foldseek/pdb_db"
echo "  ./scripts/compute_diversity_novelty.sh outputs/proteins/\${RUN_NAME}"
