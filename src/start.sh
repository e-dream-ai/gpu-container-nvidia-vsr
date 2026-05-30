#!/usr/bin/env bash

TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"backend:cudaMallocAsync"}

python3 - <<'EOF'
import subprocess, sys

result = subprocess.run(
    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
    capture_output=True, text=True,
)
print(f"nvidia-vsr: NVIDIA driver version: {result.stdout.strip()}")

try:
    import torch
    from nvvfx import VideoSuperRes

    torch.cuda.set_device(0)
    sr = VideoSuperRes(device=0, quality=VideoSuperRes.QualityLevel.HIGH)
    sr.input_width, sr.input_height = 960, 540
    sr.output_width, sr.output_height = 1920, 1080
    sr.load()
    print(f"nvidia-vsr: NvVFX warmup OK (loaded={sr.is_loaded})")
except Exception as e:
    print(f"nvidia-vsr: NvVFX warmup FAILED: {e}", file=sys.stderr)
    sys.exit(1)
EOF

if [ $? -ne 0 ] && [ "$SKIP_VFX_PRECHECK" != "true" ]; then
    echo "nvidia-vsr: NvVFX unavailable on this host (likely driver < 570.190); refusing job so it reschedules" >&2
    exit 1
fi

if [ "$SERVE_API_LOCALLY" == "true" ]; then
    echo "nvidia-vsr: Starting RunPod Handler (local API)"
    python3 -u /rp_handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    echo "nvidia-vsr: Starting RunPod Handler"
    python3 -u /rp_handler.py
fi
