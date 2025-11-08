# Quick Start Guide

This guide will help you get CodeAssist running quickly.

## Prerequisites

✅ **Python 3.10+** - The repository was tested with Python 3.12.3  
✅ **Docker** - Required for running the application containers  
✅ **UV Package Manager** - For dependency management  

## Installation

### 1. Install UV Package Manager

If you don't have UV installed, you can install it via pip:

```bash
pip install uv
```

Or use the official installer:
```bash
# macOS
brew install uv

# Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Verify Docker is Running

Make sure Docker is installed and running:

```bash
docker --version
docker ps
```

## Running CodeAssist

### Basic Usage

To run CodeAssist with all features enabled:

```bash
uv run run.py
```

When you first run it, you'll be prompted for a HuggingFace token. You can get one from:
https://huggingface.co/settings/tokens

### Common Options

Run without telemetry and model uploads (for testing):
```bash
uv run run.py --no-telemetry --no-upload
```

Run on a different port (if 3000 is occupied):
```bash
uv run run.py --port 3001
```

Get help and see all available options:
```bash
uv run run.py --help
```

## First Run

On the first run, UV will:
1. Download and install all Python dependencies (~3-5 minutes)
2. The application will start and prompt for your HuggingFace token
3. Docker containers will be pulled and started
4. Your browser should open automatically to http://localhost:3000

## Troubleshooting

### Port Already in Use
If port 3000 is already allocated, use a different port:
```bash
uv run run.py --port 3001
```

### Docker Not Running
Ensure Docker Desktop (or Docker daemon) is running before starting CodeAssist.

### Dependencies Installation
The first run takes longer because UV needs to download PyTorch and other large dependencies. Subsequent runs will be much faster.

## Next Steps

Once running, refer to the main [README.md](README.md) for:
- How to use the Web UI
- Training your model
- Tips & tricks for better results
