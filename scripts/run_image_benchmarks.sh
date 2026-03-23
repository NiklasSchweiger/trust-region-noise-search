#!/bin/bash
# ==============================================================================
# Text-to-Image Benchmark Suite (DrawBench)
#
# Runs TRS (our method) + random_search + zero_order + base (pure sampling)
# across SDXL Lightning and Stable Diffusion with HPSv2 and ImageReward.
#
# SLURM: Adjust SLURM_PARTITION, SLURM_QOS, and the activate_env() function
# below before submitting.  Without SLURM, copy the Python command from any
# heredoc and run it directly.
#
# Run from repo root: ./scripts/run_image_benchmarks.sh
# ==============================================================================
set -e

# ------------------------------------------------------------------------------
# Configuration – edit these to reproduce paper experiments
# ------------------------------------------------------------------------------
MODALITY=image
PIPELINES=("sdxl_lightning" "sd")
REWARDS=("hps" "image_reward")
BENCHMARKS=("draw_bench")
SOLVERS=("base" "trs" "random_search" "zero_order")
SEED=42

# Experiment budget / solver parameters
# Matches the paper configuration
BATCH_SIZE=20
NUM_REGIONS=15
NUM_ITERATIONS=20
WARMUP_BATCHES=4
ORACLE_BUDGET=400
CHECKPOINTS="[100,200,300,400]" # NOTE: TRS can be run with checkpoints, but one needs to adjust the warm-up batches to match the paper results. 


# ------------------------------------------------------------------------------
# SLURM configuration – adjust to your cluster
# ------------------------------------------------------------------------------
SLURM_PARTITION="gpu"           
SLURM_QOS=""                    
SLURM_TIME="2-00:00:00"
SLURM_GRES="gpu:1"

activate_env() {
    # Replace with however you activate your Python environment
    # Examples:
    #   conda activate trns
    #   source /path/to/venv/bin/activate
    #   micromamba activate trns
    micromamba activate trns
}

# ------------------------------------------------------------------------------
# Setup (run from repo root so batch_logs and noise_optimization are found)
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
mkdir -p "${REPO_ROOT}/batch_logs"

SLURM_QOS_LINE=""
[[ -n "$SLURM_QOS" ]] && SLURM_QOS_LINE="#SBATCH --qos=$SLURM_QOS"

echo -e "${BLUE}=== Image Benchmarks ===${NC}"
echo "Pipelines: ${PIPELINES[*]}  Rewards: ${REWARDS[*]}  Solvers: ${SOLVERS[*]}"
echo "Seed: $SEED"
echo ""

JOB_COUNT=0

for PIPELINE in "${PIPELINES[@]}"; do
for REWARD in "${REWARDS[@]}"; do
for BENCHMARK in "${BENCHMARKS[@]}"; do
for SOLVER in "${SOLVERS[@]}"; do

    WANDB_PROJECT="IMAGE_BENCHMARK_${PIPELINE^^}_${REWARD^^}"
    LOG_DIR="${REPO_ROOT}/batch_logs/IMAGE_${PIPELINE}_${REWARD}"
    mkdir -p "$LOG_DIR"

    # Base: pure model sampling (single forward pass, no optimisation)
    if [[ "$SOLVER" == "base" ]]; then
        job_name="draw_bench_base_${PIPELINE}_${REWARD}"
        SOLVER_ARGS="solver=random_search solver.batch_size=1 solver.num_iterations=1"
        BUDGET_JOB=1
        CHECKPOINTS_JOB="[1]"
    else
        job_name="draw_bench_${SOLVER}_${PIPELINE}_${REWARD}"
        SOLVER_ARGS="solver=$SOLVER solver.batch_size=$BATCH_SIZE solver.num_iterations=$NUM_ITERATIONS"
        if [[ "$SOLVER" == "trs" ]]; then
            SOLVER_ARGS="$SOLVER_ARGS solver.num_regions=$NUM_REGIONS solver.warmup_batches=$WARMUP_BATCHES solver.tr.update_factor=1.75 solver.tr.init_length=0.6"
        fi
        BUDGET_JOB=$ORACLE_BUDGET
        CHECKPOINTS_JOB=$CHECKPOINTS
    fi

    JOB_COUNT=$((JOB_COUNT + 1))
    echo -e "${YELLOW}[$JOB_COUNT] $job_name${NC}"

    sbatch --job-name="$job_name" \
           --output="$LOG_DIR/${job_name}.%j.out" \
           --error="$LOG_DIR/${job_name}.%j.err" <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=$SLURM_PARTITION
$SLURM_QOS_LINE
#SBATCH --gres=$SLURM_GRES
#SBATCH --time=$SLURM_TIME

cd "$REPO_ROOT"
find noise_optimization -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

$(declare -f activate_env)
activate_env

echo "Starting: $job_name  Solver: $SOLVER  Pipeline: $PIPELINE  Reward: $REWARD"

srun --unbuffered python -m noise_optimization.main \
    modality=$MODALITY \
    model=$PIPELINE \
    benchmark=$BENCHMARK \
    reward_function=$REWARD \
    $SOLVER_ARGS \
    oracle_budget=$BUDGET_JOB \
    +seed=$SEED \
    wandb_project=$WANDB_PROJECT \
    wandb_name=$job_name \
    wandb=true \
    logging.wandb.scaling.enabled=true \
    logging.wandb.scaling.budget_checkpoints=$CHECKPOINTS_JOB \
    +save_best_structures=false

echo "Completed: $job_name"
EOF

    echo -e "${GREEN}✓ Submitted: $job_name${NC}"

done; done; done; done

echo ""
echo -e "${BLUE}Submitted $JOB_COUNT jobs.${NC}"
echo "Logs: ${REPO_ROOT}/batch_logs/"
