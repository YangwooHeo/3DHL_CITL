"""Persistent isolated CPU camera preprocessing, separate from the Torch process.

Some Windows Torch/SciPy installations load incompatible OpenMP runtimes into
one process. This worker isolates calibration and registration without disabling
runtime checks or paying process-startup/calibration cost on every iteration.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import traceback
import uuid

import numpy as np

from citl_camera_preprocess import CameraPreprocessConfig, ProcessedCameraFrame


class CameraPreprocessorWorker:
    """CameraPreprocessor interface backed by one subprocess for an entire run.

    ``log_path`` names the worker stderr log. A private response directory is
    created alongside that explicit path and removed on close. Raw captures are
    only read; they are never modified or deleted. Calls must be sequential.
    """

    def __init__(self, config: CameraPreprocessConfig, log_path: Path | str,
                 *, timeout_s: float = 900):
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Camera worker timeout_s must be finite and positive")
        self.config = config
        self.log_path = Path(log_path).expanduser().resolve()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_root = self.log_path.parent / (".camera-worker-" + uuid.uuid4().hex)
        self._output_root.mkdir()
        self._process = None
        self._log = None
        self._reader = None
        self._responses = queue.Queue()
        self._request_id = 0
        self._timeout_s = float(timeout_s)
        self._closed = False
        try:
            self._log = self.log_path.open("w", encoding="utf-8")
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self._process = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
                 "--output-root", str(self._output_root)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=flags,
            )
            self._reader = threading.Thread(target=self._read_responses, daemon=True,
                                            name="citl-camera-worker-responses")
            self._reader.start()
            response = self._request("init", config=asdict(config))
            self.exposures_us = tuple(response["exposures_us"])
            self.raw_shape = tuple(response["raw_shape"])
            self.output_size = int(response["output_size"])
            self.provenance = response["provenance"]
            self.provenance["worker"] = {"pid": self._process.pid,
                "isolated_from_torch": True, "log_path": str(self.log_path)}
        except BaseException:
            self.close()
            raise

    def _read_responses(self):
        try:
            for line in self._process.stdout:
                self._responses.put(line)
        finally:
            self._responses.put(None)

    def _diagnostic(self, message):
        code = self._process.poll() if self._process is not None else None
        try:
            tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-6000:].strip()
        except OSError:
            tail = "(worker log unavailable)"
        return RuntimeError(f"{message}; camera worker exit code={code}; log={self.log_path}"
                            + (f"\n{tail}" if tail else ""))

    def _request(self, operation, **payload):
        if self._closed or self._process is None:
            raise RuntimeError("Camera preprocessing worker is closed")
        self._request_id += 1
        message = {"id": self._request_id, "operation": operation, **payload}
        try:
            self._process.stdin.write(json.dumps(message, default=str, allow_nan=False) + "\n")
            self._process.stdin.flush()
            line = self._responses.get(timeout=self._timeout_s)
            if line is None:
                # Reap first so even a native crash has a deterministic exit code.
                self._process.wait(timeout=2)
                raise self._diagnostic("Camera preprocessing worker terminated unexpectedly")
            try:
                response = json.loads(line)
            except (TypeError, json.JSONDecodeError) as exc:
                raise self._diagnostic(f"Invalid camera worker protocol response: {line[:200]!r}") from exc
            if response.get("id") != self._request_id:
                raise self._diagnostic("Camera worker response ID mismatch")
            if not response.get("ok"):
                raise self._diagnostic("Camera preprocessing failed: " + response.get("error", "unknown error"))
            return response
        except queue.Empty as exc:
            error = self._diagnostic(f"Camera preprocessing timed out after {self._timeout_s:g}s")
            self.close()
            raise error from exc
        except (BrokenPipeError, OSError) as exc:
            error = self._diagnostic("Communication with camera preprocessing worker failed")
            self.close()
            raise error from exc
        except BaseException:
            self.close()
            raise

    def process(self, target_paths, reference_paths, stem) -> ProcessedCameraFrame:
        response = self._request("process", stem=str(stem),
            target_paths={str(key): str(Path(path).expanduser().resolve()) for key, path in target_paths.items()},
            reference_paths={str(key): str(Path(path).expanduser().resolve()) for key, path in reference_paths.items()})
        result_path = self._output_root / f"result-{response['id']}.npz"
        try:
            with np.load(result_path, allow_pickle=False) as result:
                return ProcessedCameraFrame(
                    intensity=result["intensity"].copy(),
                    valid_mask=result["valid_mask"].copy(),
                    confidence=result["confidence"].copy(),
                    fallback_mask=result["fallback_mask"].copy(),
                    diagnostics=json.loads(str(result["diagnostics"].item())),
                )
        except BaseException:
            self.close()
            raise
        finally:
            result_path.unlink(missing_ok=True)

    def close(self):
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is not None:
            if process.stdin is not None:
                try:
                    process.stdin.close()  # EOF asks an idle worker to exit normally.
                except OSError:
                    pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            if self._reader is not None:
                self._reader.join(timeout=1)
            if process.stdout is not None:
                process.stdout.close()
        if self._log is not None:
            self._log.close()
        # Only remove our generated result files from our unique owned directory.
        if self._output_root.is_dir():
            for path in self._output_root.glob("result-*.npz"):
                path.unlink(missing_ok=True)
            self._output_root.rmdir()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback_value):
        self.close()
        return False


def _worker_main(output_root: Path):
    output_root = output_root.resolve(strict=True)
    if not output_root.is_dir():
        raise ValueError("Parent must supply an existing private worker output directory")
    # No module imported by this worker imports Torch. Numerical backend loading
    # and all its progress output remain entirely in this child process.
    from citl_camera_preprocess import CameraPreprocessor
    processor = None
    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            request_id = request["id"]
            if not isinstance(request_id, int) or isinstance(request_id, bool) or request_id < 1:
                raise ValueError("Worker request ID must be a positive integer")
            with redirect_stdout(sys.stderr):
                if request["operation"] == "init":
                    if processor is not None:
                        raise RuntimeError("Camera worker has already been initialized")
                    processor = CameraPreprocessor(CameraPreprocessConfig(**request["config"]))
                    response = {"exposures_us": processor.exposures_us,
                                "raw_shape": processor.raw_shape,
                                "output_size": processor.output_size,
                                "provenance": processor.provenance}
                elif request["operation"] == "process":
                    if processor is None:
                        raise RuntimeError("Camera worker has not been initialized")
                    frame = processor.process(
                        {int(key): Path(path) for key, path in request["target_paths"].items()},
                        {int(key): Path(path) for key, path in request["reference_paths"].items()},
                        request["stem"],
                    )
                    np.savez(output_root / f"result-{request_id}.npz",
                             intensity=frame.intensity, valid_mask=frame.valid_mask,
                             confidence=frame.confidence, fallback_mask=frame.fallback_mask,
                             diagnostics=np.asarray(json.dumps(frame.diagnostics, default=str)))
                    response = {}
                else:
                    raise ValueError(f"Unknown camera worker operation {request['operation']!r}")
            print(json.dumps({"id": request_id, "ok": True, **response}, default=str, allow_nan=False), flush=True)
        except BaseException as exc:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({"id": request_id, "ok": False,
                              "error": f"{type(exc).__name__}: {exc}"}), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    _worker_main(arguments.output_root)
