#!/bin/bash
set -e

cd preprocessing

# Clone the repository if it doesn't exist
if [ ! -d "ClimSim" ]; then
  git clone https://github.com/leap-stc/ClimSim.git
else
  echo "ClimSim repository already exists, skipping clone."
fi

echo "Running make_folders.py"
python make_folders.py
echo "Running rawdata2npy_train.py"
python rawdata2npy_train.py
echo "Running rawdata2npy_valid.py"
python rawdata2npy_valid.py
echo "Running generate_dataset_v3_inputs.py"
python generate_dataset_v3_inputs.py
echo "Running generate_dataset_v3_outputs.py"
python generate_dataset_v3_outputs.py
echo "Running generate_dataset_v4_inputs.py"
python generate_dataset_v4_inputs.py
echo "Running generate_dataset_v4_outputs.py"
python generate_dataset_v4_outputs.py
echo "Running process_test.py"
python process_test.py
