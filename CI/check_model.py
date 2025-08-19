import argparse

import akida
from cnn2snn import convert
from quantizeml import load_model

from compute_device import compute_min_device


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
    parser.add_argument("--model", required=True, help="Path to model file (.h5 or .onnx)")
    args = parser.parse_args()

    process_model(args.model)

