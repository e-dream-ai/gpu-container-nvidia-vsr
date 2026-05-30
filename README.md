# gpu-container-nvidia-vsr

RunPod serverless container for NVIDIA Video Super Resolution upscaling.

## How It Works

Calls the NVIDIA VFX SDK (`nvvfx.VideoSuperRes`) directly. The source video is streamed frame-by-frame with PyAV, each frame is super-resolved on the GPU, and the result is re-encoded with the hardware HEVC encoder
(`hevc_nvenc`, falling back to `libx265`) and uploaded to Cloudflare R2.

## Job Input

```json
{
    "input": {
        "video_url": "https://...",
        "scale": 2,
        "quality": "HIGH"
    }
}
```

- `video_url` (required): http(s) URL of the source video.
- `scale` (optional, default `2`): one of `1`, `2`, `3`, `4`. `1` denoises/deblurs at the same resolution.
- `quality` (optional, default `HIGH`): `LOW`, `MEDIUM`, `HIGH`, or `ULTRA`.

## GPU Compatibility

Uses the NVIDIA VFX SDK, which supports datacenter GPUs (not limited to consumer
RTX). Requires driver **570.190+**.

| GPU          | Supported | Recommended            |
| ------------ | --------- | ---------------------- |
| L40 / L40S   | Yes       | **Primary target**     |
| RTX 6000 Ada | Yes       | Good                   |
| A100 / H100  | Yes       | Overkill for upscaling |
| RTX 4090     | Yes       | Works                  |

## Quality Settings

| Setting | Speed    | Quality   |
| ------- | -------- | --------- |
| LOW     | Fastest  | Basic     |
| MEDIUM  | Fast     | Good      |
| HIGH    | Moderate | Very good |
| ULTRA   | Slowest  | Best      |

## Build

```bash
docker build -t edreamai/gpu-container-nvidia-vsr:latest .
```

## Algorithm

`infinidream_algorithm: "nvidia-uprez"`
