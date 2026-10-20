# Parse Akida Models with Rust

Load and parse any `.fbz` model from this zoo using pure Rust — no Python required.

## Quick Start

```bash
# 1. Clone rustChip
git clone https://github.com/syntheticChemistry/rustChip.git

# 2. Download a model (requires Git LFS)
cd ../models
git lfs pull --include="vision/classification/mobilenetv1/**/*.fbz"

# 3. Parse it
cd ../rustChip
cargo run --example parse_fbz -- ../models/vision/classification/mobilenetv1/plant_village/*.fbz
```

## How It Works

The `.fbz` format is **Snappy-compressed FlatBuffers**:

```
[varint: uncompressed_size] [snappy_chunks] [zero_padding]
```

rustChip's parser:
1. Reads the Snappy varint header (the first bytes are NOT magic — they're the uncompressed size)
2. Decompresses the Snappy stream, handling BrainChip's trailing zero padding
3. Parses the resulting FlatBuffer into layer descriptions, weight tensors, and metadata

## Validated Models

All models from this zoo parse correctly with rustChip (tested April 2026, re-validated October 2026):

| Domain | Models tested | Status |
|--------|--------------|--------|
| Vision (v1) | akidanet, mobilenet, vgg, yolo, gxnor, ds_cnn | ✅ |
| Vision (v2) | akidanet, mobilenet, vgg, yolo, gxnor, centernet | ✅ |

## Programmatic API

```rust
use akida_models::Model;

// Load and parse
let model = Model::load("path/to/model.fbz")?;

// Inspect
println!("Layers: {}", model.layer_count());
println!("Parameters: {}", model.param_count());
println!("Input shape: {:?}", model.input_shape());

// Deploy to hardware (requires AKD1000/AKD1500)
let device = akida_driver::DeviceManager::discover()?.open_first()?;
model.load_to_device(&device)?;
```

## Links

- [syntheticChemistry/rustChip](https://github.com/syntheticChemistry/rustChip) — the Rust driver and model parser
- [FBZ parser fix discussion](https://github.com/Brainchip-Inc/akida_dw_edma/pull/20) — how the format was reverse-engineered
