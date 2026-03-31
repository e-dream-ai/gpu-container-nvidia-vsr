# gpu-container-nvidia-vsr

RunPod serverless container for Nvidia RTX Video Super Resolution upscaling via ComfyUI.

## How It Works

Uses the `RTXVideoSuperResolution` node from [Nvidia_RTX_Nodes_ComfyUI](https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI). This runs on the Nvidia VFX SDK — no model files needed.

## GPU Compatibility

Uses Nvidia VFX SDK which supports datacenter GPUs (not limited to consumer RTX):

| GPU | Supported | Recommended |
|-----|-----------|-------------|
| L40S | Yes | **Primary target** |
| A40 | Yes | **Good fallback** |
| A100 | Yes | Overkill for upscaling |
| H100 | Yes | Overkill for upscaling |
| RTX 4090 | Yes | Works |

## Quality Settings

| Setting | Speed | Quality |
|---------|-------|---------|
| LOW | Fastest | Basic |
| MEDIUM | Fast | Good |
| HIGH | Moderate | Very good |
| ULTRA | Slowest | Best |

## Build

```bash
docker build -t edream/gpu-container-nvidia-vsr:latest .
```

## Algorithm

`infinidream_algorithm: "nvidia-uprez"`

## Workflows

- `test_input.json` — single image upscale (2x, ULTRA)
- `test_input_video.json` — video upscale with audio preservation

## Status

**First cut** — RTX node identified, GPU compatibility confirmed. Waiting on Jef
for his preferred quality settings and workflow details.
