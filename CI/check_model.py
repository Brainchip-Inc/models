import sys
import os
import argparse

import akida
from cnn2snn import convert
from tensorflow.keras.models import load_model
import onnx

sys.path.append(os.path.join(os.path.dirname(__file__), "CI"))
from compute_device import compute_min_device


def load_input_model(path):
    """Load a model from .h5 or .onnx file."""
    if path.endswith(".h5"):
        return load_model(path)
    elif path.endswith(".onnx"):
        return onnx.load(path)
    else:
        raise ValueError(f"Unsupported model format: {path}")


def process_model(file_path):
    try:
        print(f"Loading model: {file_path}")
        base_model = load_input_model(file_path)
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return

    try:
        model_ak = convert(base_model)
    except Exception as e:
        print(f"❌ Error converting {file_path}: {e}")
        return

    try:
        device = compute_min_device(model_ak, enable_hwpr=True)
        result = len(device.mesh.nps) // 4
        print(f"✅ {file_path}: needs {result} Akida nodes")
    except Exception as e:
        print(f"❌ Error mapping {file_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model file (.h5 or .onnx)")
    args = parser.parse_args()

    process_model(args.model)

