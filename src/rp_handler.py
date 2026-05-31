import base64
import contextlib
import os
import signal
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import boto3
import requests
import runpod
import torch
import torch.nn.functional as F
from botocore.exceptions import ClientError
from nvvfx import VideoSuperRes
from torchvision.io import encode_jpeg

GPU_DEVICE = 0
HEVC_MAX_DIMENSION = 8192
OUTPUT_BITRATE = 16_000_000
DEFAULT_FPS = 30
DOWNLOAD_TIMEOUT_SECONDS = 600
CANCEL_CHECK_INTERVAL_FRAMES = 8
PROGRESS_INTERVAL_SECONDS = 1.0
PREVIEW_INTERVAL_SECONDS = 2.0
PREVIEW_MAX_SIDE = 512
PREVIEW_JPEG_QUALITY = 85
CODEC_CANDIDATES = ("hevc_nvenc", "libx265")
WORK_DIR = Path(os.environ.get("VSR_WORK_DIR", "/tmp/vsr"))
REFRESH_WORKER = os.environ.get("REFRESH_WORKER", "false").lower() == "true"


class JobCancelled(Exception):
    """Raised when a cancellation signal interrupts frame processing."""


class CancellationToken:
    def __init__(self) -> None:
        self.cancelled = False

    def install_signal_handlers(self) -> None:
        try:
            signal.signal(signal.SIGINT, self._on_signal)
            signal.signal(signal.SIGTERM, self._on_signal)
        except ValueError:
            print("nvidia-vsr - not on main thread, cancellation signals unavailable")

    def _on_signal(self, signum: int, _frame: object) -> None:
        print(f"nvidia-vsr - received signal {signum}, requesting cancellation")
        self.cancelled = True

    def is_set(self) -> bool:
        return self.cancelled


class ProgressReporter:
    def __init__(self, job: dict) -> None:
        self._job = job
        self._start = 0.0
        self._last_progress_at = 0.0
        self._last_preview_at = 0.0

    def start(self) -> None:
        self._start = time.perf_counter()
        self._last_progress_at = 0.0
        self._last_preview_at = 0.0

    def report(self, processed: int, total: int, frame: torch.Tensor) -> None:
        now = time.perf_counter()
        if now - self._last_progress_at < PROGRESS_INTERVAL_SECONDS:
            return
        self._last_progress_at = now

        data: dict = {}
        if total > 0 and processed > 0:
            data["progress"] = round(min(99.9, processed / total * 100.0), 1)
            elapsed = now - self._start
            data["countdown_ms"] = int(elapsed / processed * (total - processed) * 1000)

        if now - self._last_preview_at >= PREVIEW_INTERVAL_SECONDS:
            preview = self._encode_preview(frame)
            if preview:
                data["preview_frame"] = preview
                self._last_preview_at = now

        if data:
            runpod.serverless.progress_update(self._job, data)

    def _encode_preview(self, frame: torch.Tensor) -> str | None:
        """Downscale a (3, H, W) RGB tensor to a small base64 JPEG."""
        try:
            _, height, width = frame.shape
            scale = PREVIEW_MAX_SIDE / max(height, width)
            if scale < 1.0:
                frame = F.interpolate(frame.unsqueeze(0), scale_factor=scale, mode="area").squeeze(0)
            image = (frame.clamp(0.0, 1.0) * 255.0).byte().cpu()
            jpeg = encode_jpeg(image, quality=PREVIEW_JPEG_QUALITY)
            return base64.b64encode(jpeg.numpy().tobytes()).decode("utf-8")
        except Exception as error:
            print(f"nvidia-vsr - preview encode failed: {error}")
            return None


@dataclass(frozen=True)
class JobParams:
    video_url: str
    scale: int
    quality: str


@dataclass(frozen=True)
class UpscaleStats:
    frames: int
    input_resolution: tuple[int, int]
    output_resolution: tuple[int, int]
    encoder: str
    elapsed_seconds: float


def validate_input(job_input: object) -> tuple[JobParams | None, str | None]:
    if not isinstance(job_input, dict):
        return None, "Input must be a JSON object"

    video_url = job_input.get("video_url")
    if not isinstance(video_url, str) or not video_url.startswith(("http://", "https://")):
        return None, "'video_url' must be an http(s) URL"

    scale = job_input.get("scale", 2)
    if scale not in (1, 2, 3, 4):
        return None, "'scale' must be one of 1, 2, 3, 4"

    quality = str(job_input.get("quality", "HIGH")).upper()
    valid_qualities = set(VideoSuperRes.QualityLevel.__members__)
    if quality not in valid_qualities:
        return None, f"'quality' must be one of: {', '.join(sorted(valid_qualities))}"

    return JobParams(video_url=video_url, scale=int(scale), quality=quality), None


def download_video(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                file.write(chunk)


def _frame_to_gpu_tensor(frame: av.VideoFrame) -> torch.Tensor:
    arr = frame.to_ndarray(format="rgb24")
    tensor = torch.from_numpy(arr).to(f"cuda:{GPU_DEVICE}")
    return (tensor.permute(2, 0, 1).float() / 255.0).contiguous()


def _tensor_to_frame(tensor: torch.Tensor) -> av.VideoFrame:
    frame_np = (tensor.clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).contiguous().cpu().numpy()
    return av.VideoFrame.from_ndarray(frame_np, format="rgb24")


def _open_output_container(
    output_path: Path, width: int, height: int, frame_rate: Fraction
) -> tuple[av.container.OutputContainer, av.video.stream.VideoStream, str]:
    for name in CODEC_CANDIDATES:
        container = av.open(str(output_path), mode="w")
        try:
            stream = container.add_stream(name, rate=frame_rate)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            stream.bit_rate = OUTPUT_BITRATE
            stream.codec_context.open()
        except Exception:
            container.close()
            continue
        return container, stream, name

    raise RuntimeError(f"No usable H.265 encoder (tried {', '.join(CODEC_CANDIDATES)})")


def _count_total_frames(input_stream: av.video.stream.VideoStream, container_duration: int | None, fps: float) -> int:
    if input_stream.frames and input_stream.frames > 0:
        return input_stream.frames
    duration_seconds = 0.0
    if input_stream.duration and input_stream.time_base:
        duration_seconds = float(input_stream.duration * input_stream.time_base)
    elif container_duration:
        duration_seconds = container_duration / av.time_base
    return round(duration_seconds * fps) if duration_seconds > 0 else 0


def upscale_video(
    input_path: Path,
    output_path: Path,
    scale: int,
    quality: str,
    should_cancel: Callable[[], bool],
    reporter: ProgressReporter | None = None,
) -> UpscaleStats:
    torch.cuda.set_device(GPU_DEVICE)
    stream_ptr = torch.cuda.current_stream().cuda_stream

    input_container = av.open(str(input_path))
    input_stream = input_container.streams.video[0]
    input_stream.thread_type = "AUTO"

    input_width = input_stream.codec_context.width
    input_height = input_stream.codec_context.height
    output_width = input_width * scale
    output_height = input_height * scale
    fps = float(input_stream.average_rate) if input_stream.average_rate else DEFAULT_FPS
    total_frames = _count_total_frames(input_stream, input_container.duration, fps)

    print(
        f"nvidia-vsr - {input_width}x{input_height} -> {output_width}x{output_height} "
        f"@ {fps:.2f}fps, scale={scale}x, quality={quality}"
    )

    if output_width > HEVC_MAX_DIMENSION or output_height > HEVC_MAX_DIMENSION:
        input_container.close()
        raise RuntimeError(
            f"Output {output_width}x{output_height} exceeds HEVC maximum "
            f"of {HEVC_MAX_DIMENSION}x{HEVC_MAX_DIMENSION}"
        )

    sr = VideoSuperRes(device=GPU_DEVICE, quality=VideoSuperRes.QualityLevel[quality])
    sr.input_width = input_width
    sr.input_height = input_height
    sr.output_width = output_width
    sr.output_height = output_height
    sr.load()

    frame_rate = Fraction(fps).limit_denominator(10000)
    output_container, video_stream, encoder = _open_output_container(
        output_path, output_width, output_height, frame_rate
    )
    print(f"nvidia-vsr - encoder: {encoder}")

    start_time = time.perf_counter()
    if reporter:
        reporter.start()
    processed = 0
    try:
        for frame in input_container.decode(input_stream):
            if processed % CANCEL_CHECK_INTERVAL_FRAMES == 0 and should_cancel():
                raise JobCancelled(f"cancelled after {processed} frames")

            rgb_input = _frame_to_gpu_tensor(frame)
            output = sr.run(rgb_input, stream_ptr=stream_ptr)
            rgb_output = torch.from_dlpack(output.image).clone()

            for packet in video_stream.encode(_tensor_to_frame(rgb_output)):
                output_container.mux(packet)
            processed += 1

            if reporter:
                reporter.report(processed, total_frames, rgb_output)

        for packet in video_stream.encode(None):
            output_container.mux(packet)
    finally:
        output_container.close()
        input_container.close()

    elapsed = time.perf_counter() - start_time
    print(f"nvidia-vsr - processed {processed} frames in {elapsed:.1f}s ({processed / elapsed:.1f} fps)")

    return UpscaleStats(
        frames=processed,
        input_resolution=(input_width, input_height),
        output_resolution=(output_width, output_height),
        encoder=encoder,
        elapsed_seconds=elapsed,
    )


def upload_to_r2(job_id: str, file_path: Path) -> dict:
    endpoint_url = os.environ.get("R2_ENDPOINT_URL")
    access_key_id = os.environ.get("R2_ACCESS_KEY_ID")
    secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket_name = os.environ.get("R2_BUCKET_NAME")
    upload_directory = os.environ.get("R2_UPLOAD_DIRECTORY", "").strip().strip("/")
    expires_in = int(os.environ.get("R2_PRESIGNED_EXPIRY", "86400"))
    public_url_base = os.environ.get("R2_PUBLIC_URL_BASE")

    if not all([endpoint_url, access_key_id, secret_access_key, bucket_name]):
        raise RuntimeError("Missing R2 configuration")

    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name="auto",
        config=boto3.session.Config(s3={"addressing_style": "path"}),
    )

    s3_key = f"{job_id}-{file_path.name}"
    if upload_directory:
        s3_key = f"{upload_directory}/{s3_key}"

    try:
        with open(file_path, "rb") as file:
            s3_client.upload_fileobj(file, bucket_name, s3_key, ExtraArgs={"ContentType": "video/mp4"})
    except ClientError as error:
        raise RuntimeError(f"Failed to upload to R2: {error}") from error

    try:
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": s3_key},
            ExpiresIn=expires_in,
        )
    except Exception:
        if public_url_base:
            url = f"{public_url_base.rstrip('/')}/{s3_key}"
        else:
            account_id = endpoint_url.split("://")[1].split(".")[0]
            url = f"https://{account_id}.r2.dev/{s3_key}"

    return {"url": url, "s3_key": s3_key, "bucket": bucket_name}


def handler(job: dict) -> dict:
    params, error = validate_input(job.get("input"))
    if error:
        return {"error": error}

    token = CancellationToken()
    token.install_signal_handlers()

    job_id = str(job.get("id", uuid.uuid4().hex))
    work_dir = WORK_DIR / job_id
    input_path = work_dir / "input.mp4"
    output_path = work_dir / f"{job_id}-upscaled.mp4"

    try:
        download_video(params.video_url, input_path)
    except requests.RequestException as error:
        return {"error": f"Failed to download video: {error}"}

    try:
        stats = upscale_video(
            input_path,
            output_path,
            scale=params.scale,
            quality=params.quality,
            should_cancel=token.is_set,
            reporter=ProgressReporter(job),
        )
    except JobCancelled as cancelled:
        print(f"nvidia-vsr - {cancelled}")
        _cleanup(input_path, output_path)
        return {"status": "cancelled", "message": str(cancelled)}
    except Exception as error:
        print(f"nvidia-vsr - processing failed: {error}")
        _cleanup(input_path, output_path)
        return {"error": f"Upscale failed: {error}"}

    try:
        upload = upload_to_r2(job_id, output_path)
    except Exception as error:
        _cleanup(input_path, output_path)
        return {"error": str(error)}

    _cleanup(input_path, output_path)

    return {
        "status": "success",
        "download_url": upload["url"],
        "video": upload["url"],
        "s3_key": upload["s3_key"],
        "bucket": upload["bucket"],
        "frames": stats.frames,
        "input_resolution": f"{stats.input_resolution[0]}x{stats.input_resolution[1]}",
        "output_resolution": f"{stats.output_resolution[0]}x{stats.output_resolution[1]}",
        "encoder": stats.encoder,
        "refresh_worker": REFRESH_WORKER,
    }


def _cleanup(*paths: Path) -> None:
    for path in paths:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
