# File: run_p0_experiment.sh
#!/bin/bash
"""
Script to run P0 foundation experiment
"""

echo "=========================================="
echo "SLM vs LLM P0 Foundation Experiment"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "Error: Python $required_version or higher is required (found $python_version)"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check for API keys
if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Warning: No API keys found. Running in mock mode."
    echo "Set OPENAI_API_KEY or ANTHROPIC_API_KEY for real results."
fi

# Run the experiment
echo "Starting P0 experiment..."
python slm_agentic_foundation.py

echo "Experiment complete! Check ./results/ for outputs."