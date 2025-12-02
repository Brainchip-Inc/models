import argparse
import os
import subprocess

# Configure TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import akida
from cnn2snn import convert
from quantizeml import load_model
from compute_device import compute_min_device
from filtered_stream import filtered_output


def process_model(file_path):
    try:       
        print(f"Loading model: {file_path}")
        base_model = load_model(file_path)
    except Exception as e:
        raise RuntimeError(f"❌ Error loading {file_path}: {e}") from e

    try:
        model_ak = convert(base_model)
    except Exception as e:
        raise RuntimeError(f"❌ Error converting {file_path}: {e}") from e

    try:
        device = compute_min_device(model_ak, enable_hwpr=True)
        result = len(device.mesh.nps) // 4
        print(f"✅ {file_path}: needs {result} Akida nodes")
    except Exception as e:
        raise RuntimeError(f"❌ Error mapping {file_path}: {e}") from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", help="Path to model files (.h5 or .onnx)")    
    args = parser.parse_args()

    # Process each model
    with filtered_output():
        for model_file in args.models:
            if os.path.exists(model_file):
                # Move this step of the YAML workflow into the Python script
                # Download the file via Git LFS if necessary
                # git lfs pull --include="$f" --exclude=""
                subprocess.run(["git", "lfs", "pull", 
                    "--include", model_file,"--exclude", ""], 
                    capture_output=True)                
                process_model(model_file)    

