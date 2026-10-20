"""Capture three RealSense color views for a bounded check; no robot APIs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import time


def collect(pipeline, seconds, rs, np):
    # All pipelines are already running before workers begin collecting.
    warmup_end = time.monotonic() + 1.0
    deadline = warmup_end + seconds
    arrivals, numbers = [], []
    last_image = None
    while time.monotonic() < deadline:
        frames = pipeline.poll_for_frames()
        frame = frames.get_color_frame() if frames else None
        if not frame:
            time.sleep(0.005)
            continue
        now = time.monotonic()
        if now < warmup_end:
            continue
        number = frame.get_frame_number()
        if numbers and number == numbers[-1]:
            continue
        arrivals.append(now)
        numbers.append(number)
        last_image = np.asanyarray(frame.get_data()).copy()
    if last_image is None or len(numbers) < 2:
        raise RuntimeError("Camera did not produce a continuing color stream.")
    fps = (len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
    gaps = sum(max(0, b - a - 1) for a, b in zip(numbers, numbers[1:]))
    return last_image, {
        "received_frames": len(numbers),
        "observed_fps": round(fps, 2),
        "frame_number_gaps": gaps,
        "largest_arrival_gap_ms": round(max(b - a for a, b in zip(arrivals, arrivals[1:])) * 1000, 1),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true", help="Read actual cameras and save snapshots.")
    parser.add_argument("--seconds", type=int, default=5, choices=range(2, 16), metavar="2..15")
    args = parser.parse_args(argv)
    if not args.capture:
        print("PREVIEW: no camera opened. Use --capture to check three simultaneous 640x480 @ 15fps color streams.")
        return 0

    import cv2
    import numpy as np
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = list(ctx.query_devices())
    if len(devices) != 3:
        raise RuntimeError("Expected three RealSense cameras, found %d." % len(devices))
    devices.sort(key=lambda dev: dev.get_info(rs.camera_info.serial_number))
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = Path(__file__).resolve().parent / "camera_checks" / stamp
    output.mkdir(parents=True)
    pipelines, records, panels = [], [], []
    report = {
        "captured_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "mode": "three simultaneous color streams; no depth stream or robot commands",
        "requested_profile": {"width": 640, "height": 480, "fps": 15, "format": "bgr8"},
        "sample_seconds": args.seconds,
        "cameras": records,
    }
    try:
        for index, device in enumerate(devices, 1):
            serial = device.get_info(rs.camera_info.serial_number)
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
            pipeline = rs.pipeline(ctx)
            pipeline.start(config)
            pipelines.append(pipeline)
            records.append({
                "index": index,
                "serial": serial,
                "name": device.get_info(rs.camera_info.name),
                "usb": device.get_info(rs.camera_info.usb_type_descriptor),
            })
            print("Started camera %d: %s" % (index, serial), flush=True)

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(collect, pipe, args.seconds, rs, np) for pipe in pipelines]
            for record, future in zip(records, futures):
                try:
                    view, stats = future.result()
                    filename = "camera_%d.jpg" % record["index"]
                    if not cv2.imwrite(str(output / filename), view):
                        raise RuntimeError("Could not write snapshot.")
                    record.update(stats)
                    record["snapshot"] = filename
                    record["ok"] = True
                    panel = np.zeros((520, 640, 3), dtype=np.uint8)
                    panel[40:] = view
                    cv2.putText(panel, "Camera %d | %s | %.2f fps" % (record["index"], record["serial"], record["observed_fps"]),
                                (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
                    panels.append(panel)
                except Exception as exc:
                    record.update(ok=False, error=str(exc))
    finally:
        for pipeline in reversed(pipelines):
            try:
                pipeline.stop()
            except Exception as exc:
                report.setdefault("stop_errors", []).append(str(exc))
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if panels:
        if not cv2.imwrite(str(output / "all_views.jpg"), np.hstack(panels)):
            raise RuntimeError("Could not write overview.")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("OUTPUT:", output, flush=True)
    return 0 if len(records) == 3 and all(r.get("ok") for r in records) and not report.get("stop_errors") else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ImportError, RuntimeError, OSError) as exc:
        print("Camera check failed:", exc, file=sys.stderr)
        sys.exit(1)
