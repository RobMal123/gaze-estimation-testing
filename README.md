# MobileGaze Gaze Estimation Setup

This project sets up and runs gaze estimation on videos using the [MobileGaze](https://github.com/yakhyo/gaze-estimation) repository.

## Quick Start

### 1. Clone and Setup

The repository has already been cloned. To set up dependencies, run:

```bash
python setup.py
```

This will install:

- Dependencies from `gaze-estimation/requirements.txt`
- Additional packages: `opencv-python`, `matplotlib`, `torch`, `torchvision`

### 2. Download Model Weights

Download the model weights you want to use. For example, to download ResNet-50 weights:

```bash
cd gaze-estimation
sh download.sh resnet50
cd ..
```

Available models:

- `resnet18` - 43 MB, MAE: 12.84
- `resnet34` - 81.6 MB, MAE: 11.33
- `resnet50` - 91.3 MB, MAE: 11.34
- `mobilenetv2` - 9.59 MB, MAE: 13.07
- `mobileone_s0` - 4.8 MB, MAE: 12.58

Or download manually from the [releases page](https://github.com/yakhyo/gaze-estimation/releases/tag/v0.0.1) and place them in `gaze-estimation/weights/`.

### 3. Prepare Your Video

Place your test video file (e.g., `ad_test.mp4`) in the `input/` folder.

### 4. Run Gaze Estimation

Run the gaze estimation on your video:

```bash
python run_gaze_estimation.py
```

This will:

- Use `input/ad_test.mp4` as the source video (default)
- Use `resnet50` model (default)
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
├── gaze-estimation/          # Cloned MobileGaze repository
│   ├── weights/              # Model weights (download separately)
│   ├── inference.py         # Main inference script
│   └── ...
├── input/                    # Place your test videos here
├── results/                  # Output videos saved here
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

## Original Repository

For more information about MobileGaze, visit:

- GitHub: https://github.com/yakhyo/gaze-estimation
- Pre-trained models are available in the [releases](https://github.com/yakhyo/gaze-estimation/releases/tag/v0.0.1)
