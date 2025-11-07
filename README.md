# MobileGaze Gaze Estimation Setup

This project runs **inference-only** gaze estimation on videos using a streamlined version of [MobileGaze](https://github.com/yakhyo/gaze-estimation) with enhanced 3D-style arrow visualization.

## Features

✨ **Enhanced 3D-Style Arrows** - Gaze360-inspired solid arrows with depth effects and lighting
🎯 **Inference-Only** - Clean, production-ready codebase focused solely on video processing
🚀 **Optimized Dependencies** - Minimal requirements with only essential packages

## Quick Start

### 1. Clone and Setup

The repository has already been cloned. To set up dependencies, run:

```bash
python setup.py
```

This will install the core dependencies:

- `numpy`, `opencv-python`, `pillow`
- `torch`, `torchvision` (PyTorch for model inference)
- `uniface` (RetinaFace for face detection)

### 2. Download Model Weights

#### Option A: Download from Google Drive (Recommended)

Download the pre-trained ResNet-50 weights:

- **[Download resnet50.pt from Google Drive](https://drive.google.com/file/d/1iXMWdS9HwRDW7OLKN-LGr7N1ghgfT59k/view?usp=sharing)**

After downloading, place the `resnet50.pt` file in the `gaze-estimation/weights/` folder.

### 3. Prepare Your Video

Place your test video file (e.g., `test.mp4`) in the `input/` folder.

### 4. Run Gaze Estimation

Run the gaze estimation on your video:

```bash
python run_gaze_estimation.py --source input/drottninggatan.mp4 --output output/drottninggatan_out.mp4
```

This will:

- Use `input/test.mp4` as the source video (default)
- Use `resnet50` model (default)
- Save output to `output/test_out.mp4`
- Save output to `results/attention.mp4` (default)

## Custom Usage

### Command-Line Arguments

```bash
python run_gaze_estimation.py [OPTIONS]
```

Options:

- `--source PATH` - Path to source video (default: `input/ad_test.mp4`)
- `--model MODEL` - Model architecture: `resnet18`, `resnet34`, `resnet50`, `mobilenetv2`, `mobileone_s0` (default: `resnet50`)
- `--weight PATH` - Path to model weights file (default: `gaze-estimation/weights/{model}.pt`)
- `--output PATH` - Path to save output video (default: `results/attention.mp4`)
- `--dataset DATASET` - Dataset name for configuration (default: `gaze360`)
- `--view` - Display inference results in a window during processing
- `--no-view` - Do not display results (default behavior)

### Examples

```bash
# Use defaults (resnet50 on input/ad_test.mp4, no window)
python run_gaze_estimation.py

# Use a different video and model
python run_gaze_estimation.py --source input/my_video.mp4 --model resnet34

# Show results in window while processing
python run_gaze_estimation.py --source input/delivery.mp4 --view

# Custom output path
python run_gaze_estimation.py --source input/delivery.mp4 --output output/delivery_out.mp4

# Use lighter MobileNet model
python run_gaze_estimation.py --model mobilenetv2
```

## Project Structure

```
.
├── gaze-estimation/          # Streamlined MobileGaze (inference only)
│   ├── models/               # Model architectures (ResNet, MobileNet, MobileOne)
│   ├── utils/
│   │   └── helpers.py       # 3D-style arrow rendering & utilities
│   ├── weights/              # Model weights (download separately)
│   ├── config.py            # Dataset configurations
│   ├── inference.py         # Main inference script
│   └── requirements.txt     # Minimal dependencies
├── input/                    # Place your test videos here
├── output/                   # Output videos saved here
├── setup.py                 # Setup script for dependencies
├── run_gaze_estimation.py   # Wrapper script to run gaze estimation
└── README.md                # This file
```

## Troubleshooting

### Weights Not Found

If you get an error about missing weights:

1. Make sure you've downloaded the weights to `gaze-estimation/weights/`
2. The weight filename should match the model name (e.g., `resnet50.pt` for `resnet50` model)

### CUDA Issues

The script automatically uses CUDA if available, otherwise falls back to CPU. The device is automatically detected by the inference script.

### Video Not Found

Make sure your video file is in the `input/` folder and the filename matches what you specify with `--source`.

## Credits & Attribution

This project is based on [MobileGaze](https://github.com/yakhyo/gaze-estimation) by [yakhyo](https://github.com/yakhyo), streamlined for inference with enhanced visualization.

- **Original Repository**: https://github.com/yakhyo/gaze-estimation
- **License**: See `gaze-estimation/LICENSE`
- **Pre-trained models**: Available in the [releases](https://github.com/yakhyo/gaze-estimation/releases/tag/v0.0.1)

The `gaze-estimation/` folder contains the MobileGaze codebase with modifications. The wrapper scripts (`run_gaze_estimation.py`, `setup.py`) and project documentation are original contributions.
