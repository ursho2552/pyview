#!/bin/bash

# PyView Environment Setup Script
# This script creates a conda environment for PyView with all necessary dependencies

echo "Setting up PyView conda environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^pyview "; then
    echo "PyView environment already exists."
    read -p "Do you want to update it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Updating existing environment..."
        conda env update -f environment.yml
    else
        echo "Using existing environment. To activate:"
        echo "   conda activate pyview"
        exit 0
    fi
else
    # Create the environment
    echo "Creating conda environment from environment.yml..."
    conda env create -f environment.yml
fi

# Check if environment creation/update was successful
if [ $? -eq 0 ]; then
    echo "Environment ready!"

    # Activate environment and set up JupyterLab extensions
    echo "Setting up JupyterLab widget extensions..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate pyview

    # Ensure JupyterLab widgets are properly installed
    jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build
    jupyter lab build --minimize=False

    echo ""
    echo "PyView setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Activate the environment:"
    echo "   conda activate pyview"
    echo ""
    echo "2. Start Jupyter Lab:"
    echo "   jupyter lab"
    echo ""
    echo "3. Test PyView:"
    echo "   python -c \"from pyview import pyview; print('PyView ready!')\""
    echo ""
    echo "4. Open test.ipynb to see PyView in action"
    echo ""
    echo "Tip: The environment includes development tools (black, pytest) for contributing"
else
    echo "Error: Environment setup failed"
    echo "Please check the error messages above and try again"
    exit 1
fi
