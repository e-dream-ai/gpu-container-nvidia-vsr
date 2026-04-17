#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"backend:cudaMallocAsync"}

# Log driver version and pre-warm NvVFX TRT engine
python3 - <<'EOF'
import subprocess, sys

result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True)
driver = result.stdout.strip()
print(f"runpod-worker-comfy: NVIDIA driver version: {driver}")

# nvidia-vfx 0.1.0.1 requires driver >= 570.190 on Linux
try:
    major, minor, patch = (int(x) for x in driver.split("."))
    ver = (major, minor, patch)
    if ver < (570, 190, 0):
        print(f"runpod-worker-comfy: WARNING - driver {driver} may be too old for nvidia-vfx (needs 570.190+). NvVFX may fail.", file=sys.stderr)
except Exception:
    pass

# Pre-warm NvVFX to trigger TRT engine compilation before first job
try:
    import nvvfx
    print(f"runpod-worker-comfy: nvvfx location: {nvvfx.__file__}")
    with nvvfx.VideoSuperRes("HIGH") as sr:
        sr.output_width = 1920
        sr.output_height = 1080
        sr.load()
        print("runpod-worker-comfy: NvVFX warmup OK")
except Exception as e:
    print(f"runpod-worker-comfy: NvVFX warmup FAILED: {e}", file=sys.stderr)
EOF

# Serve the API and don't shutdown the container
if [ "$SERVE_API_LOCALLY" == "true" ]; then
    echo "runpod-worker-comfy: Starting ComfyUI"
    python3 /comfyui/main.py --disable-auto-launch --disable-metadata --listen &

    echo "runpod-worker-comfy: Starting RunPod Handler"
    python3 -u /rp_handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    echo "runpod-worker-comfy: Starting ComfyUI"
    python3 /comfyui/main.py --disable-auto-launch --disable-metadata &

    echo "runpod-worker-comfy: Starting RunPod Handler"
    python3 -u /rp_handler.py
fi