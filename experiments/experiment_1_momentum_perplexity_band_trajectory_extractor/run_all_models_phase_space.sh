#!/bin/bash

# Shell script to run momentum_perplexity_phase_space_experiment.py for multiple models
# This script systematically tests all models with configurable parameters
# Supports both existing text analysis and token generation modes
# All parameters can be overridden via environment variables

set -e  # Exit on any error

# =============================================================================
# CONFIGURATION SECTION - Modify these parameters as needed
# =============================================================================

# Models file configuration
MODELS_FILE="${MODELS_FILE:-test_models.txt}"

# Example usage with environment variables:
# GENERATE_TOKENS="10" ./run_all_models_phase_space.sh              # Generate 10 tokens per trajectory
# DEVICE="cpu" NUM_ARTICLES=5 ./run_all_models_phase_space.sh       # Use CPU with 5 articles
# PERPLEXITY_CUTOFF="50" ./run_all_models_phase_space.sh            # Limit perplexity to 50

# Read models from file
if [ ! -f "$MODELS_FILE" ]; then
    echo "Error: Models file not found: $MODELS_FILE"
    exit 1
fi

# Load models from file into array
MODELS=()
while IFS= read -r model; do
    # Skip empty lines and comments
    if [[ -n "$model" && ! "$model" =~ ^[[:space:]]*# ]]; then
        MODELS+=("$model")
    fi
echo "model: $model"
done < "$MODELS_FILE"

# Experiment parameters - customize as needed (can be overridden via environment variables)
DEVICE="${DEVICE:-mps}"                            # Device to use: "cpu", "cuda", or "mps"
NUM_ARTICLES="${NUM_ARTICLES:-20}"                 # Number of Wikipedia articles to analyze
MAX_STEPS_PER_TEXT="${MAX_STEPS_PER_TEXT:-10}"     # Max trajectory steps per text
PERPLEXITY_CUTOFF="${PERPLEXITY_CUTOFF:-}"         # Maximum perplexity for visualization (empty = no cutoff)
GENERATE_TOKENS="${GENERATE_TOKENS:-}"             # Number of tokens to generate per trajectory (empty = use existing text only)

# Logging configuration
LOG_DIR="experiment_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/run_all_models_phase_space_${TIMESTAMP}.log"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Function to print colored output
print_header() {
    echo -e "\n\033[1;34m════════════════════════════════════════════════════════════════════════════════\033[0m"
    echo -e "\033[1;34m$1\033[0m"
    echo -e "\033[1;34m════════════════════════════════════════════════════════════════════════════════\033[0m\n"
}

print_info() {
    echo -e "\033[1;32m[INFO]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

# Function to estimate time remaining
estimate_time() {
    local completed=$1
    local total=$2
    local start_time=$3
    local current_time=$(date +%s)
    local elapsed=$((current_time - start_time))
    
    if [ $completed -gt 0 ]; then
        local avg_time_per_model=$((elapsed / completed))
        local remaining_models=$((total - completed))
        local estimated_remaining=$((avg_time_per_model * remaining_models))
        local hours=$((estimated_remaining / 3600))
        local minutes=$(((estimated_remaining % 3600) / 60))
        echo "${hours}h ${minutes}m"
    else
        echo "Calculating..."
    fi
}

# Function to create sanitized model name for directories
sanitize_model_name() {
    echo "$1" | sed 's/[^a-zA-Z0-9._-]/_/g'
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

print_header "MOMENTUM-PERPLEXITY PHASE SPACE EXPERIMENT - MULTI-MODEL RUN"

# Create log directory
mkdir -p "$LOG_DIR"

# Log all output to file as well as console
exec > >(tee -a "$MAIN_LOG") 2>&1

print_info "Starting multi-model phase space experiment run at $(date)"
print_info "Logs will be saved to: $MAIN_LOG"
print_info "Models file: $MODELS_FILE"
print_info "Total models to test: ${#MODELS[@]}"

# Validate that models were loaded
if [ ${#MODELS[@]} -eq 0 ]; then
    print_error "No models found in $MODELS_FILE"
    print_error "Make sure the file exists and contains valid model names"
    exit 1
fi

# Display loaded models for verification
print_info "Loaded models from $MODELS_FILE:"
for i in "${!MODELS[@]}"; do
    echo "  $((i+1)). ${MODELS[$i]}"
done

# Display configuration
print_info "CONFIGURATION:"
echo "  Device: $DEVICE"
echo "  Articles per model: $NUM_ARTICLES"
echo "  Max steps per text: $MAX_STEPS_PER_TEXT"
echo "  Perplexity cutoff: ${PERPLEXITY_CUTOFF:-'None'}"
echo "  Generate tokens: ${GENERATE_TOKENS:-'None (use existing text only)'}"

# Validate that the experiment script exists
SCRIPT_PATH="momentum_perplexity_phase_space_experiment.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    print_error "Experiment script not found: $SCRIPT_PATH"
    print_error "Make sure you're running this script from the correct directory"
    exit 1
fi

# Check if conda environment is active and get the correct Python path
if [[ "$CONDA_DEFAULT_ENV" == "" ]]; then
    print_error "No conda environment is active. Please activate the conda environment first:"
    print_error "conda activate llm_dynamics"
    exit 1
fi

# Use the Python from the active conda environment
PYTHON_CMD="$CONDA_PREFIX/bin/python"
if [ ! -f "$PYTHON_CMD" ]; then
    print_error "Python not found in conda environment: $PYTHON_CMD"
    print_error "Current conda environment: $CONDA_DEFAULT_ENV"
    exit 1
fi

print_info "Using Python from conda environment '$CONDA_DEFAULT_ENV': $PYTHON_CMD"

# Quick check that required packages are available
print_info "Checking for required packages..."
if ! "$PYTHON_CMD" -c "import torch, transformers, numpy, datasets" &> /dev/null; then
    print_error "Required packages not found in conda environment. Please install them:"
    print_error "pip install torch transformers numpy datasets matplotlib seaborn"
    exit 1
fi
print_info "All required packages found"

# Start timing
START_TIME=$(date +%s)
SUCCESSFUL_RUNS=0
FAILED_RUNS=0
FAILED_MODELS=()

# Main loop through models
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_NUM=$((i + 1))
    SANITIZED_MODEL=$(sanitize_model_name "$MODEL")
    
    print_header "MODEL $MODEL_NUM/${#MODELS[@]}: $MODEL"
    
    # Create model-specific log file
    MODEL_LOG="${LOG_DIR}/model_${MODEL_NUM}_${SANITIZED_MODEL}_phase_space_${TIMESTAMP}.log"
    
    print_info "Starting phase space experiment for model: $MODEL"
    print_info "Model log: $MODEL_LOG"
    
    # Estimate time remaining
    TIME_REMAINING=$(estimate_time $i ${#MODELS[@]} $START_TIME)
    print_info "Estimated time remaining: $TIME_REMAINING"
    
    # Build the command with all parameters
    CMD="$PYTHON_CMD $SCRIPT_PATH"
    CMD="$CMD --model_name \"$MODEL\""
    CMD="$CMD --device $DEVICE"
    CMD="$CMD --num_articles $NUM_ARTICLES"
    CMD="$CMD --max_steps_per_text $MAX_STEPS_PER_TEXT"
    
    # Add perplexity cutoff if specified
    if [ -n "$PERPLEXITY_CUTOFF" ]; then
        CMD="$CMD --perplexity_cutoff $PERPLEXITY_CUTOFF"
    fi
    
    # Add generate tokens if specified
    if [ -n "$GENERATE_TOKENS" ]; then
        CMD="$CMD --generate_tokens $GENERATE_TOKENS"
    fi
    
    print_info "Command: $CMD"
    
    # Record start time for this model
    MODEL_START_TIME=$(date +%s)
    
    # Run the experiment with timeout and error handling
    print_info "Executing experiment..."
    
    # Run the experiment directly without timeout
    SUCCESS=false
    
    if $PYTHON_CMD $SCRIPT_PATH --model_name "$MODEL" --device $DEVICE --num_articles $NUM_ARTICLES --max_steps_per_text $MAX_STEPS_PER_TEXT ${PERPLEXITY_CUTOFF:+--perplexity_cutoff $PERPLEXITY_CUTOFF} ${GENERATE_TOKENS:+--generate_tokens $GENERATE_TOKENS} > "$MODEL_LOG" 2>&1; then
        SUCCESS=true
    fi
    
    MODEL_END_TIME=$(date +%s)
    MODEL_DURATION=$((MODEL_END_TIME - MODEL_START_TIME))
    MINUTES=$((MODEL_DURATION / 60))
    SECONDS=$((MODEL_DURATION % 60))
    
    if [ "$SUCCESS" = true ]; then
        print_success "Model $MODEL completed successfully in ${MINUTES}m ${SECONDS}s"
        SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
        
        # Find the output directory created by the experiment
        OUTPUT_DIRS=$(find . -maxdepth 1 -name "phase_space_analysis_*" -type d -newer "$MODEL_LOG" 2>/dev/null | head -1)
        if [ -n "$OUTPUT_DIRS" ]; then
            # Rename output directory to include model name
            NEW_DIR_NAME="${OUTPUT_DIRS}_${SANITIZED_MODEL}"
            if [ "$OUTPUT_DIRS" != "$NEW_DIR_NAME" ]; then
                mv "$OUTPUT_DIRS" "$NEW_DIR_NAME" 2>/dev/null || true
                print_info "Results saved to: $NEW_DIR_NAME"
            else
                print_info "Results saved to: $OUTPUT_DIRS"
            fi
        fi
    else
        print_error "Model $MODEL failed after ${MINUTES} minutes"
        print_error "Check log file for details: $MODEL_LOG"
        FAILED_RUNS=$((FAILED_RUNS + 1))
        FAILED_MODELS+=("$MODEL")
        
        # Show last few lines of error log
        print_warning "Last 10 lines of error log:"
        tail -n 10 "$MODEL_LOG" | sed 's/^/    /'
    fi
    
    # Progress update
    COMPLETED=$((SUCCESSFUL_RUNS + FAILED_RUNS))
    print_info "Progress: $COMPLETED/${#MODELS[@]} models completed ($SUCCESSFUL_RUNS successful, $FAILED_RUNS failed)"
    
    # Short break between models (except for the last one)
    if [ $MODEL_NUM -lt ${#MODELS[@]} ]; then
        print_info "Waiting 5 seconds before next model..."
        sleep 5
    fi
done

# =============================================================================
# FINAL SUMMARY
# =============================================================================

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

print_header "EXPERIMENT SUMMARY"

print_info "Completed at: $(date)"
print_info "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
print_info "Models tested: ${#MODELS[@]}"
print_success "Successful runs: $SUCCESSFUL_RUNS"

if [ $FAILED_RUNS -gt 0 ]; then
    print_error "Failed runs: $FAILED_RUNS"
    print_error "Failed models:"
    for failed_model in "${FAILED_MODELS[@]}"; do
        echo "  - $failed_model"
    done
fi

# List generated result directories
print_info "Generated result directories:"
find . -maxdepth 1 -name "phase_space_analysis_*" -type d | sort | while read dir; do
    echo "  - $dir"
done

print_info "All logs saved in: $LOG_DIR"
print_info "Main log: $MAIN_LOG"

# Exit with appropriate code
if [ $FAILED_RUNS -eq 0 ]; then
    print_success "All phase space experiments completed successfully!"
    exit 0
else
    print_warning "Some experiments failed. Check individual model logs for details."
    exit 1
fi 