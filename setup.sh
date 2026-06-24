#!/bin/bash
# setup.sh
set -e  # stop script si erreur

echo "============================"
echo " Streamlit ML Setup Script "
echo "============================"

echo "Python version: $(python --version)"

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "Verifying installations..."

python -c "
import streamlit
import joblib
import sklearn
import pandas
import numpy
print('✅ Core ML stack OK')

echo ""
echo "Listing installed ML packages:"
pip list | grep -E "streamlit|joblib|scikit-learn|xgboost|pandas|numpy"

echo ""
echo "Setup complete successfully!"






