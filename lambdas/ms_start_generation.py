import os, json, time, uuid, boto3
import io
import numpy as np
from PIL import Image

# When the Agent decides to call POST /synthesis, 
# Bedrock invokes the mapped Lambda (ms-start-generation) 
# with an event that contains the action’s parameters (e.g., run_id, source_key, …).
# Your Lambda reads those parameters, does the work (e.g., start a job, write to S3), 
# and returns a JSON payload the Agent can show/use.

s3 = boto3.client("s3")
BUCKET = os.environ.get("MS_BUCKET", "medimage-digitaltwin")

def handler(event, context):
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    recipe = {}
    input_key = None
    source_key = None
    from_run_id = None
    export_png = False

    try:
        if event and "body" in event and event["body"]:
            body = event["body"]
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except Exception:
                    body = {}
            recipe = body.get("recipe", {})
            input_key = body.get("input_key")
            source_key = body.get("source_key")
            from_run_id = body.get("run_id")
            export_png = bool(body.get("export_png", False))
    except Exception:
        pass

    # Determine source data key with precedence: run_id -> source_key -> input_key -> default raw
    resolved_key = None
    if from_run_id:
        resolved_key = f"work/{from_run_id}/processed.npy"
    elif source_key:
        resolved_key = source_key
    elif input_key:
        resolved_key = input_key
    else:
        resolved_key = os.environ.get("MS_RAW_KEY", "raw/chestmnist.npz")

    # Load input from S3 (supports .npz with common keys, or .npy)
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=resolved_key)
        body_bytes = obj["Body"].read()
        data = None
        if resolved_key.endswith(".npz"):
            npz = np.load(io.BytesIO(body_bytes))
            for k in ["images", "train_images", "val_images", "test_images"]:
                if k in npz:
                    data = npz[k]
                    break
            if data is None:
                raise KeyError("Could not find images array in npz file")
        elif resolved_key.endswith(".npy"):
            data = np.load(io.BytesIO(body_bytes))
        else:
            raise ValueError("Unsupported input format. Use .npy or .npz")
    except Exception as e:
        err = {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "error": f"Failed to read input from s3://{BUCKET}/{resolved_key}",
                "details": str(e)
            }, ensure_ascii=False)
        }
        return err

    # Generate a simple "digital twin": preserve global distribution, remove per-sample identifiers
    # Strategy: For each image, sample from a Gaussian with the image's mean/std to create a synthetic twin
    try:
        if data.ndim == 3:  # (N, H, W)
            means = data.mean(axis=(1,2), keepdims=True)
            stds = data.std(axis=(1,2), keepdims=True) + 1e-6
            rng = np.random.default_rng()
            twin = rng.normal(loc=means, scale=stds, size=data.shape).astype(data.dtype)
        elif data.ndim == 2:  # (H, W)
            mean = data.mean()
            std = data.std() + 1e-6
            rng = np.random.default_rng()
            twin = rng.normal(loc=mean, scale=std, size=data.shape).astype(data.dtype)
        else:
            raise ValueError("Unsupported data shape; expected (N,H,W) or (H,W)")
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "error": "Digital twin generation failed",
                "details": str(e)
            }, ensure_ascii=False)
        }

    # Persist outputs and status
    twin_key = f"work/{run_id}/digital_twin.npy"
    s3.put_object(
        Bucket=BUCKET,
        Key=twin_key,
        Body=twin.tobytes(),
        ContentType="application/octet-stream"
    )

    status_key = f"work/{run_id}/status.json"
    status_obj = {
        "status": "completed",
        "recipe": recipe,
        "input_key": input_key,
        "source_key": source_key,
        "from_run_id": from_run_id,
        "digital_twin_key": twin_key
    }
    s3.put_object(
        Bucket=BUCKET, Key=status_key,
        Body=json.dumps(status_obj, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json"
    )

    # Optional PNG export (preview up to 32 images)
    import zipfile
    
    if export_png:
        zip_key = f"work/{run_id}/synthetic.zip"
        try:
            arr = twin
            if arr.ndim == 2:
                arr = arr[None, ...]
            count = min(arr.shape[0], 32)

            # Build a ZIP in memory
            zipbuf = io.BytesIO()
            with zipfile.ZipFile(zipbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for idx in range(count):
                    img = arr[idx]
                    # scale to 0-255 for PNG
                    img_min = float(img.min())
                    img_max = float(img.max())
                    denom = (img_max - img_min) if (img_max - img_min) > 1e-12 else 1.0
                    img_uint8 = ((img - img_min) / denom * 255.0).clip(0, 255).astype("uint8")
                    pil = Image.fromarray(img_uint8)

                    # encode PNG into bytes
                    png_bytes = io.BytesIO()
                    pil.save(png_bytes, format="PNG")
                    png_bytes.seek(0)

                    # write into the zip under a stable name
                    zf.writestr(f"{idx:04d}.png", png_bytes.read())

            # upload the zip
            zipbuf.seek(0)
            s3.put_object(
                Bucket=BUCKET,
                Key=zip_key,
                Body=zipbuf.getvalue(),
                ContentType="application/zip",
                # Optional: force download with a nice filename in browsers
                ContentDisposition=f'attachment; filename="{run_id}_synthetic.zip"'
            )

            # update status with zip path
            status_obj["digital_twin_zip"] = zip_key
            s3.put_object(
                Bucket=BUCKET, Key=status_key,
                Body=json.dumps(status_obj, ensure_ascii=False).encode("utf-8"),
                ContentType="application/json"
            )
        except Exception:
            # Best-effort: ignore ZIP failures so the main flow still succeeds
            pass


    msg = f"Generated digital twin for {resolved_key}. Output at s3://{BUCKET}/{twin_key}"
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "run_id": run_id,
            "message": msg,
            "digital_twin_key": twin_key,
            "status_key": status_key
        }, ensure_ascii=False)
    }