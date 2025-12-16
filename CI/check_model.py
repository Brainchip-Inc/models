import argparse

import akida
from cnn2snn import convert
from quantizeml import load_model


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
        device = akida.compute_min_device(model_ak, enable_hwpr=True)
        result = len(device.mesh.nps) // 4
        print(f"✅ {file_path}: needs {result} Akida nodes")
    except Exception as e:
        raise RuntimeError(f"❌ Error mapping {file_path}: {e}") from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="Path to model files (.h5 or .onnx)")
    args = parser.parse_args()

    # Process each model
    for model_file in args.models:
        process_model(model_file)
