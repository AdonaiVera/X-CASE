#!/bin/bash

# Improved Simulation Script
# Runs only scenario 1 (the only available scenario) with different models
# Models: qwen, gpt4o, claude, deepseek

# Generate a unique session ID for this run
SESSION_ID=$RANDOM
SESSION_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "🚀 Starting Improved Simulation Script"
echo "======================================"
echo "Session ID: $SESSION_ID"
echo "Session Timestamp: $SESSION_TIMESTAMP"
echo "Running scenario 1 with different models"
echo "Available models: qwen, gpt4o, claude, deepseek"
echo ""

# Define models to test
MODELS=("qwen" "gpt4o" "claude" "deepseek")
SCENARIO_INDEX=1  # Only scenario 1 exists
STEPS=600

# Function to update model config
update_model_config() {
    local model=$1
    echo "🔧 Updating model configuration to: $model"
    
    # Create a temporary Python script to update the model config
    python3 -c "
import sys
sys.path.append('.')
from model_config import set_multimodal_model
set_multimodal_model('$model')
print(f'✅ Model configuration updated to: $model')
"
}

# Function to run simulation for a specific model
run_simulation_for_model() {
    local model=$1
    local random_id=$RANDOM
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local target_name="simulation_scenario1_${model}_${random_id}_${timestamp}"
    
    echo ""
    echo "🎯 Running simulation for model: $model"
    echo "Target name: $target_name"
    echo "Random ID: $random_id"
    echo "Timestamp: $timestamp"
    echo "Scenario index: $SCENARIO_INDEX"
    echo "Steps: $STEPS"
    echo "----------------------------------------"
    
    # Update model configuration
    update_model_config "$model"
    
    # Run the simulation
    ./run_backend_automatic.sh \
        --env_name simulacra \
        -o base_party \
        -t "$target_name" \
        -s $STEPS \
        --ui True \
        --scenario_index $SCENARIO_INDEX
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ Simulation completed successfully for model: $model"
    else
        echo "❌ Simulation failed for model: $model (exit code: $exit_code)"
    fi
    
    echo "Finished simulation for model: $model"
    echo ""
}

# Main execution
echo "📋 Starting simulations for ${#MODELS[@]} models..."
echo ""

for model in "${MODELS[@]}"; do
    run_simulation_for_model "$model"
done

echo "🎉 All simulations completed!"
echo "======================================"
echo "Summary:"
echo "- Session ID: $SESSION_ID"
echo "- Session Timestamp: $SESSION_TIMESTAMP"
echo "- Scenario: 1 (only available scenario)"
echo "- Models tested: ${MODELS[*]}"
echo "- Steps per simulation: $STEPS"
echo ""
echo "Check the logs/ directory for detailed results."
echo "Each simulation has a unique random ID and timestamp."
