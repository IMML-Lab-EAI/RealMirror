#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ISAACSIM_PY="${ISAACSIM_PY:-/isaac-sim/python.sh}"

OUTPUT_DIR="runs/eval_all_smolvla/$(date +%Y%m%d_%H%M%S)"
NUM_ROLLOUTS_OVERRIDE=""
MAX_HORIZON_OVERRIDE=""
HEADLESS=1
SAVE_TRAJECTORY_VIDEO=1
TRAJECTORY_RECORDING=1
CONTINUE_ON_ERROR=0
RESUME_RUN=""
RESUME_LATEST=0
OUTPUT_DIR_EXPLICIT=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Run all five RealMirror benchmark tasks with SmolVLA.

Usage:
  bash script/eval_all_smolvla.sh [options]

Options:
  --output-dir PATH       Common evaluation output root
  --resume-run PATH       Resume an existing all-task run directory
  --resume-latest         Resume the latest run under runs/eval_all_smolvla
  --num-rollouts N        Override the official rollout count for every task
  --max-horizon N         Override the official horizon for every task
  --no-headless           Launch Isaac Sim with a window
  --no-trajectory-video   Keep HDF5 trajectories but skip preview.mp4
  --disable-trajectories  Disable HDF5 trajectory recording
  --continue-on-error     Continue with the next task if one task fails
  --dry-run               Print the resume plan and commands without running Isaac Sim
  -h, --help              Show this help

Environment:
  ISAACSIM_PY              Isaac Sim Python launcher (default: /isaac-sim/python.sh)

Examples:
  bash script/eval_all_smolvla.sh
  bash script/eval_all_smolvla.sh --num-rollouts 1 --max-horizon 10
  bash script/eval_all_smolvla.sh --resume-latest
  bash script/eval_all_smolvla.sh --resume-run runs/eval_all_smolvla/20260826_102746
EOF
}

require_value() {
    if [[ $# -lt 2 || -z "$2" ]]; then
        echo "Missing value for $1" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            require_value "$@"
            OUTPUT_DIR="$2"
            OUTPUT_DIR_EXPLICIT=1
            shift 2
            ;;
        --resume-run)
            require_value "$@"
            RESUME_RUN="$2"
            shift 2
            ;;
        --resume-latest|--resume)
            RESUME_LATEST=1
            shift
            ;;
        --num-rollouts)
            require_value "$@"
            NUM_ROLLOUTS_OVERRIDE="$2"
            shift 2
            ;;
        --max-horizon)
            require_value "$@"
            MAX_HORIZON_OVERRIDE="$2"
            shift 2
            ;;
        --no-headless)
            HEADLESS=0
            shift
            ;;
        --no-trajectory-video)
            SAVE_TRAJECTORY_VIDEO=0
            shift
            ;;
        --disable-trajectories)
            TRAJECTORY_RECORDING=0
            shift
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -x "$ISAACSIM_PY" ]]; then
    echo "Isaac Sim Python launcher is not executable: $ISAACSIM_PY" >&2
    exit 1
fi

cd "$PROJECT_ROOT"

if [[ -n "$RESUME_RUN" && "$RESUME_LATEST" -eq 1 ]]; then
    echo "Use either --resume-run or --resume-latest, not both." >&2
    exit 2
fi

if [[ "$OUTPUT_DIR_EXPLICIT" -eq 1 && ( -n "$RESUME_RUN" || "$RESUME_LATEST" -eq 1 ) ]]; then
    echo "--output-dir cannot be combined with a resume option." >&2
    exit 2
fi

if [[ "$RESUME_LATEST" -eq 1 ]]; then
    RESUME_RUN="$({
        find runs/eval_all_smolvla \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            -printf '%T@ %p\n' 2>/dev/null || true
    } | sort -nr | head -n 1 | cut -d' ' -f2-)"
    if [[ -z "$RESUME_RUN" ]]; then
        echo "No previous run found under runs/eval_all_smolvla." >&2
        exit 1
    fi
fi

RESUME_MODE=0
if [[ -n "$RESUME_RUN" ]]; then
    if [[ ! -d "$RESUME_RUN" ]]; then
        echo "Resume run directory does not exist: $RESUME_RUN" >&2
        exit 1
    fi
    OUTPUT_DIR="$RESUME_RUN"
    RESUME_MODE=1
    echo "Resuming all-task evaluation from: $OUTPUT_DIR"
fi

if [[ ! -d "$PROJECT_ROOT/SmolVLM2-500M-Video-Instruct" ]]; then
    cat >&2 <<EOF
Missing SmolVLA base model:
  $PROJECT_ROOT/SmolVLM2-500M-Video-Instruct

Download it on the host into the bind-mounted RealMirror directory:
  hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct \\
      --local-dir $PROJECT_ROOT/SmolVLM2-500M-Video-Instruct
EOF
    exit 1
fi

mkdir -p "$OUTPUT_DIR/logs"

FAILED_TASKS=()

latest_task_progress_dir() {
    local task_name="$1"
    local task_root="$OUTPUT_DIR/$task_name"
    local latest_csv=""

    if [[ ! -d "$task_root" ]]; then
        return 0
    fi

    latest_csv="$({
        find "$task_root" \
            -mindepth 2 \
            -maxdepth 2 \
            -type f \
            -name evaluation_data.csv \
            -printf '%T@ %p\n' 2>/dev/null || true
    } | sort -nr | head -n 1 | cut -d' ' -f2-)"

    if [[ -n "$latest_csv" ]]; then
        dirname "$latest_csv"
    fi
}

read_progress_counts() {
    local csv_path="$1"

    awk -F',' '
        /^#/ { next }
        !header_seen {
            for (i = 1; i <= NF; i++) {
                column = $i
                gsub(/^[[:space:]\"]+|[[:space:]\"]+$/, "", column)
                if (column == "success") {
                    success_column = i
                }
            }
            header_seen = 1
            next
        }
        {
            total += 1
            value = $success_column
            gsub(/^[[:space:]\"]+|[[:space:]\"]+$/, "", value)
            if (success_column > 0 && value != "" && value != "-1") {
                completed += 1
            }
        }
        END { print total + 0, completed + 0, success_column + 0 }
    ' "$csv_path"
}

run_task() {
    local task_name="$1"
    local default_rollouts="$2"
    local default_horizon="$3"
    local area_file="$4"
    shift 4

    local num_rollouts="${NUM_ROLLOUTS_OVERRIDE:-$default_rollouts}"
    local max_horizon="${MAX_HORIZON_OVERRIDE:-$default_horizon}"
    local resume_dir=""
    local progress_csv=""
    local total_rollouts=0
    local completed_rollouts=0
    local success_column=0
    local log_mode="overwrite"

    if [[ "$RESUME_MODE" -eq 1 ]]; then
        resume_dir="$(latest_task_progress_dir "$task_name")"
        if [[ -n "$resume_dir" ]]; then
            progress_csv="$resume_dir/evaluation_data.csv"
            read -r total_rollouts completed_rollouts success_column \
                <<< "$(read_progress_counts "$progress_csv")"

            if [[ "$success_column" -eq 0 || "$total_rollouts" -eq 0 ]]; then
                echo "Invalid or empty progress CSV: $progress_csv" >&2
                exit 1
            fi

            num_rollouts="$total_rollouts"
            if [[ "$completed_rollouts" -ge "$total_rollouts" ]]; then
                echo "Skipping completed task: $task_name ($completed_rollouts/$total_rollouts rollouts)"
                return 0
            fi

            log_mode="append"
            echo "Resuming $task_name from rollout $((completed_rollouts + 1))/$total_rollouts"
            echo "Resume directory: $resume_dir"
        else
            echo "No previous progress for $task_name; starting it as a new task."
        fi
    fi

    local command=(
        "$ISAACSIM_PY"
        script/eval.py
        --task "$task_name"
        --model-type smolvla
        --arc2gear
        --num-rollouts "$num_rollouts"
        --max-horizon "$max_horizon"
        --output-dir "$OUTPUT_DIR"
    )

    if [[ -n "$resume_dir" ]]; then
        command+=(--resume-dir "$resume_dir")
    fi

    if [[ -n "$area_file" ]]; then
        command+=(--area-file "$area_file")
    fi
    if [[ "$HEADLESS" -eq 1 ]]; then
        command+=(--headless)
    fi
    if [[ "$SAVE_TRAJECTORY_VIDEO" -eq 0 ]]; then
        command+=(--no-trajectory-video)
    fi
    if [[ "$TRAJECTORY_RECORDING" -eq 0 ]]; then
        command+=(--disable-trajectory-recording)
    fi
    command+=("$@")

    echo
    echo "================================================================"
    echo "Starting $task_name (rollouts=$num_rollouts, horizon=$max_horizon)"
    echo "================================================================"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf 'Command:'
        printf ' %q' "${command[@]}"
        printf '\n'
        return 0
    fi

    local log_path="$OUTPUT_DIR/logs/${task_name}.log"
    local tee_args=("$log_path")
    if [[ "$log_mode" == "append" ]]; then
        tee_args=(-a "$log_path")
    fi

    if "${command[@]}" 2>&1 | tee "${tee_args[@]}"; then
        echo "Completed $task_name"
    else
        local exit_code=${PIPESTATUS[0]}
        echo "Failed $task_name (exit code: $exit_code)" >&2
        FAILED_TASKS+=("$task_name")
        if [[ "$CONTINUE_ON_ERROR" -eq 0 ]]; then
            exit "$exit_code"
        fi
    fi
}

run_task \
    Task1_Kitchen_Cleanup \
    400 \
    400 \
    data/eval/data_area/data_area_task1.txt

run_task \
    Task2_Cup_to_Cup_Transfer \
    200 \
    320 \
    data/eval/data_area/data_area_task2.txt

run_task \
    Task3_Assembly_Line_Sorting \
    100 \
    3000 \
    ""

run_task \
    Task4_Can_Stacking \
    400 \
    500 \
    data/eval/data_area/data_area_task4.txt \
    --use-stability-check \
    --stability-frames 100

run_task \
    Task5_Air_Fryer_Manipulation \
    400 \
    500 \
    data/eval/data_area/data_area_task5.txt

if [[ ${#FAILED_TASKS[@]} -gt 0 ]]; then
    echo "Failed tasks: ${FAILED_TASKS[*]}" >&2
    exit 1
fi

echo
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run completed. No Isaac Sim evaluation was started."
    echo "Output: $OUTPUT_DIR"
    exit 0
fi

echo "All SmolVLA evaluations completed."
echo "Output: $OUTPUT_DIR"
