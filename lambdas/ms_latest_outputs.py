import os
import io
import json
import time
import uuid
import zipfile
import boto3
import numpy as np
from PIL import Image
from typing import Tuple, Optional

# -------- Config --------
s3 = boto3.client("s3")
BUCKET = os.environ.get("MS_BUCKET", "medimage-digitaltwin")
# If no run_id/source_key provided, use this default dataset key
DEFAULT_RAW_KEY = os.environ.get("MS_RAW_KEY", "raw/chestmnist.npz")

# -------- Preprocessing ops --------
def denoise(data: np.ndarray) -> np.ndarray:
    """Mean filter (3x3) per image."""
    from scipy.ndimage import uniform_filter  # import here to keep cold start small
    if data.ndim == 3:      # (N,H,W)
        return np.stack([uniform_filter(img, size=3) for img in data], axis=0)
    elif data.ndim == 2:    # (H,W)
        return uniform_filter(data, size=3)
    else:
        raise ValueError("Unsupported data shape for denoising; expected (N,H,W) or (H,W).")

def normalize(data: np.ndarray) -> np.ndarray:
    """Min-max normalization to [0,1]."""
    dmin = float(np.min(data))
    dmax = float(np.max(data))
    denom = (dmax - dmin) if (dmax - dmin) > 1e-8 else 1.0
    return (data - dmin) / denom

def resample(data: np.ndarray, factor: int = 2) -> np.ndarray:
    """Very simple temporal/sample down-sampling by stride."""
    if data.ndim == 3:      # (N,H,W)
        return data[::factor]
    elif data.ndim == 2:    # (H,W) single image -> no-op by default
        return data
    else:
        raise ValueError("Unsupported data shape for resample; expected (N,H,W) or (H,W).")

# -------- S3 helpers --------
def s3_get_bytes(bucket: str, key: str) -> bytes:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()

def s3_put_bytes(bucket: str, key: str, data: bytes, content_type: str, *, content_disposition: Optional[str] = None) -> None:
    kwargs = dict(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    if content_disposition:
        kwargs["ContentDisposition"] = content_disposition
    s3.put_object(**kwargs)

def save_npy_to_s3(arr: np.ndarray, bucket: str, key: str) -> None:
    """Save a proper .npy (with header) so np.load works later."""
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    s3_put_bytes(bucket, key, buf.getvalue(), "application/octet-stream")

# -------- I/O & imaging --------
NPZ_IMAGE_KEYS = ["images", "train_images", "val_images", "test_images"]

def load_array_from_s3(bucket: str, key: str) -> np.ndarray:
    """Load np.ndarray from S3, supporting .npz (dict-like) or .npy."""
    body = s3_get_bytes(bucket, key)
    bio = io.BytesIO(body)
    if key.endswith(".npz"):
        npz = np.load(bio)
        for k in NPZ_IMAGE_KEYS:
            if k in npz:
                return npz[k]
        raise KeyError(f"Could not find any of {NPZ_IMAGE_KEYS} in {key}")
    elif key.endswith(".npy"):
        return np.load(bio)
    else:
        raise ValueError("Unsupported input format. Please provide .npy or .npz")

def to_uint8(img: np.ndarray) -> np.ndarray:
    """Scale any numeric image to uint8 [0,255] with min-max per-image."""
    img = np.asarray(img)
    vmin = float(img.min())
    vmax = float(img.max())
    denom = (vmax - vmin) if (vmax - vmin) > 1e-12 else 1.0
    out = ((img - vmin) / denom * 255.0).clip(0, 255).astype("uint8")
    return out

def write_pngs_to_s3_prefix(arr: np.ndarray, bucket: str, prefix: str, max_count: int = 32) -> int:
    """Save up to max_count images as PNGs to s3://bucket/prefix/0000.png, returns count saved."""
    if arr.ndim == 2:
        arr = arr[None, ...]  # (1,H,W)
    if arr.ndim != 3:
        raise ValueError("PNG export expects (N,H,W) or (H,W) arrays (grayscale).")
    n = min(arr.shape[0], max_count)
    for i in range(n):
        png_bytes = io.BytesIO()
        Image.fromarray(to_uint8(arr[i])).save(png_bytes, format="PNG")
        png_bytes.seek(0)
        s3_put_bytes(bucket, f"{prefix}{i:04d}.png", png_bytes.getvalue(), "image/png")
    return n

def make_zip_of_pngs(arr: np.ndarray, max_count: int = 32) -> bytes:
    """Return a ZIP (bytes) containing up to max_count PNGs generated from arr."""
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError("ZIP export expects (N,H,W) or (H,W) arrays (grayscale).")
    n = min(arr.shape[0], max_count)
    zipbuf = io.BytesIO()
    with zipfile.ZipFile(zipbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i in range(n):
            png_io = io.BytesIO()
            Image.fromarray(to_uint8(arr[i])).save(png_io, format="PNG")
            png_io.seek(0)
            zf.writestr(f"{i:04d}.png", png_io.read())
    zipbuf.seek(0)
    return zipbuf.getvalue()

# -------- Handler --------
def handler(event, context):
    """
    Preprocess Lambda
    Request body JSON fields (all optional):
      - run_id:      str  use digital twin of an existing run (work/<run_id>/digital_twin.npy)
      - source_key:  str  direct S3 key for input (overrides run_id); supports .npz/.npy
      - denoise:     bool apply mean filter
      - normalize:   bool min-max normalize to [0,1]
      - resample:    bool down-sample by factor (2)
      - export_png:  bool save up to 32 preview PNGs to S3
      - export_zip:  bool save a ZIP of those PNGs to S3
      - recipe:      obj  free-form metadata (stored in status.json)
    """

    # -------- Parse event body --------
    run_id = None
    recipe = {}
    denoise_flag = False
    normalize_flag = False
    resample_flag = False
    source_key = None
    export_png = False
    export_zip = False

    try:
        if event and "body" in event and event["body"]:
            body = event["body"]
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except Exception:
                    body = {}
            recipe         = body.get("recipe", {})
            denoise_flag   = bool(body.get("denoise", False))
            normalize_flag = bool(body.get("normalize", False))
            resample_flag  = bool(body.get("resample", False))
            run_id         = body.get("run_id")
            source_key     = body.get("source_key")
            export_png     = bool(body.get("export_png", False))
            export_zip     = bool(body.get("export_zip", False))
    except Exception:
        # Ignore parse errors; fields remain defaults
        pass

    # -------- Resolve input key --------
    try:
        if run_id and not source_key:
            # Use the digital twin from a previous run
            source_key = f"work/{run_id}/digital_twin.npy"
        if not source_key:
            source_key = DEFAULT_RAW_KEY

        data = load_array_from_s3(BUCKET, source_key)
    except Exception as e:
        err = {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "error": f"Failed to read input from s3://{BUCKET}/{source_key}",
                "details": str(e)
            }, ensure_ascii=False)
        }
        return err

    # -------- Apply preprocessing steps --------
    try:
        if denoise_flag:
            data = denoise(data)
        if normalize_flag:
            data = normalize(data)
        if resample_flag:
            data = resample(data, factor=2)
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "error": "Preprocessing failed",
                "details": str(e)
            }, ensure_ascii=False)
        }

    # -------- Persist outputs --------
    if not run_id:
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # Save processed as a proper .npy
    processed_key = f"work/{run_id}/processed.npy"
    try:
        save_npy_to_s3(data, BUCKET, processed_key)
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "error": "Failed to save processed.npy",
                "details": str(e)
            }, ensure_ascii=False)
        }

    # Prepare status.json
    status_key = f"work/{run_id}/status.json"
    status_obj = {
        "status": "completed",
        "recipe": recipe,
        "options": {
            "denoise": denoise_flag,
            "normalize": normalize_flag,
            "resample": resample_flag
        },
        "source_key": source_key,
        "output_key": processed_key
    }

    # Optional: export preview PNGs
    if export_png:
        try:
            preview_prefix = f"work/{run_id}/processed_png/"
            count = write_pngs_to_s3_prefix(data, BUCKET, preview_prefix, max_count=32)
            status_obj["processed_png_prefix"] = preview_prefix
            status_obj["processed_png_count"] = count
        except Exception:
            # best-effort
            pass

    # Optional: ZIP of preview PNGs
    if export_zip:
        try:
            zip_bytes = make_zip_of_pngs(data, max_count=32)
            zip_key = f"work/{run_id}/processed_preproc.zip"
            s3_put_bytes(
                BUCKET,
                zip_key,
                zip_bytes,
                "application/zip",
                content_disposition=f'attachment; filename="{run_id}_preproc.zip"'
            )
            status_obj["processed_zip_key"] = zip_key
        except Exception:
            # best-effort
            pass

    # Save status.json
    try:
        s3_put_bytes(
            BUCKET,
            status_key,
            json.dumps(status_obj, ensure_ascii=False).encode("utf-8"),
            "application/json"
        )
    except Exception:
        # Don't fail the run if status write has issues
        pass

    # -------- Response --------
    msg = f"Run {run_id} completed. Output at s3://{BUCKET}/{processed_key}"
    response = {
        "run_id": run_id,
        "message": msg,
        "options": status_obj["options"],
        "output_key": processed_key
    }
    if "processed_png_prefix" in status_obj:
        response["processed_png_prefix"] = status_obj["processed_png_prefix"]
        response["processed_png_count"] = status_obj["processed_png_count"]
    if "processed_zip_key" in status_obj:
        response["processed_zip_key"] = status_obj["processed_zip_key"]

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(response, ensure_ascii=False)
    }

# -------- Local test --------
if __name__ == "__main__":
    # Simulate an event as AWS would send
    test_event = {
        "body": json.dumps({
            "denoise": True,
            "normalize": True,
            "resample": False,
            "export_png": True,
            "export_zip": True,
            # "run_id": "existing_run_id",
            # "source_key": "raw/chestmnist.npz"
        })
    }
    print(handler(test_event, None))
