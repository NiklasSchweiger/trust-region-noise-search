#!/bin/bash
# ==============================================================================
# QM9 Multi-Property Target Benchmark
#
# Optimises six QM9 chemical properties simultaneously (alpha, gap, homo,
# lumo, mu, Cv) using TRS, random_search, and zero_order.
#
# Optional: for accurate property prediction, set up the in-repo property_prediction
# module (noise_optimization/core/models/qm9/property_prediction/); see its README.
#
# SLURM: Adjust SLURM_PARTITION, SLURM_QOS, and activate_env() below.
# Without SLURM, copy any Python command from a heredoc and run it directly.
#
# Run from repo root: ./scripts/run_qm9_target_benchmarks.sh
# ==============================================================================
set -e

MODALITY=qm9_target
BENCHMARK=qm9_properties
SOLVERS=("trs" "random_search" "zero_order")
PROPERTIES="[alpha,gap,homo,lumo,mu,Cv]"

# paper configuration
BATCH_SIZE=100
NUM_REGIONS=15
NUM_ITERATIONS=20
WARMUP_BATCHES=4
ORACLE_BUDGET=2000
CHECKPOINTS="[500,1000,1500,2000]" # NOTE: TRS can be run with checkpoints, but one needs to adjust the warm-up batches to match the paper results. 


# ------------------------------------------------------------------------------
# SLURM configuration -- adjust to your cluster
# ------------------------------------------------------------------------------
SLURM_PARTITION="gpu"
SLURM_QOS=""
SLURM_TIME="0-12:00:00"

activate_env() {
    micromamba activate trns
}

# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
mkdir -p "${REPO_ROOT}/batch_logs/QM9_TARGET"

SLURM_QOS_LINE=""
[[ -n "$SLURM_QOS" ]] && SLURM_QOS_LINE="#SBATCH --qos=$SLURM_QOS"
WANDB_PROJECT="QM9_TARGET_BENCHMARK"

echo -e "${BLUE}=== QM9 Target Benchmarks ===${NC}"
echo "Solvers: ${SOLVERS[*]}  Properties: $PROPERTIES"
echo ""

JOB_COUNT=0

for SOLVER in "${SOLVERS[@]}"; do
    JOB_COUNT=$((JOB_COUNT + 1))
    job_name="qm9_target_${SOLVER}"

    SOLVER_ARGS="solver=$SOLVER solver.batch_size=$BATCH_SIZE solver.num_iterations=$NUM_ITERATIONS"
    if [[ "$SOLVER" == "trs" ]]; then
        SOLVER_ARGS="$SOLVER_ARGS solver.num_regions=$NUM_REGIONS solver.warmup_batches=$WARMUP_BATCHES 
    fi

    echo -e "${YELLOW}[$JOB_COUNT] $job_name${NC}"

    sbatch --job-name="$job_name" \
           --output="${REPO_ROOT}/batch_logs/QM9_TARGET/${job_name}.%j.out" \
           --error="${REPO_ROOT}/batch_logs/QM9_TARGET/${job_name}.%j.err" <<HEREDOC
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

echo "Starting: $job_name  Solver: $SOLVER"

srun --unbuffered python -m noise_optimization.main \\
    modality=$MODALITY \\
    benchmark=$BENCHMARK \\
    +benchmark.properties=$PROPERTIES \\
    $SOLVER_ARGS \\
    oracle_budget=$ORACLE_BUDGET \\
    wandb_project=$WANDB_PROJECT \\
    wandb_name=$job_name \\
    wandb=true \\
    logging.wandb.scaling.enabled=true \\
    logging.wandb.scaling.budget_checkpoints=$CHECKPOINTS \\
    +save_best_structures=true

echo "Completed: $job_name"
HEREDOC

    echo -e "${GREEN}Submitted: $job_name${NC}"
done

echo ""
echo -e "${BLUE}Submitted $JOB_COUNT jobs.  Logs: ${REPO_ROOT}/batch_logs/QM9_TARGET/${NC}"
