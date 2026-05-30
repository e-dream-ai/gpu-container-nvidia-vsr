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
driver = result.stdout.strip()
print(f"nvidia-vsr: NVIDIA driver version: {driver}")

# nvidia-vfx requires driver >= 570.190 on Linux.
try:
    if tuple(int(x) for x in driver.split(".")) < (570, 190, 0):
        print(
            f"nvidia-vsr: WARNING - driver {driver} may be too old for nvidia-vfx "
            "(needs 570.190+). NvVFX may fail.",
            file=sys.stderr,
        )
except Exception:
    pass

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
EOF

if [ "$SERVE_API_LOCALLY" == "true" ]; then
    echo "nvidia-vsr: Starting RunPod Handler (local API)"
    python3 -u /rp_handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    echo "nvidia-vsr: Starting RunPod Handler"
    python3 -u /rp_handler.py
fi
