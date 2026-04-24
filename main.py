import cv2
from ultralytics import YOLO
from rtmlib import RTMPose
from boxmot.trackers.strongsort.strongsort import StrongSort
from collections import defaultdict
import random
import colorsys
import torch
import json
import sys
import time
import os

from ProcessDet import processYOLOResults
from ProcessPose import processPoseResults


class SieveAI:
    def __init__(self) -> None:
        self.pt_model = "yolo26m.pt"
        self.engine_model = "yolo26m.engine"
        self.model = self.engine_model if os.path.exists(self.engine_model) else self.pt_model
        self.openposeKpts = False
        self.captureFrames = True

        self.use_cuda = torch.cuda.is_available()
        self.DEVICE_boxmot = "0" if self.use_cuda else "cpu"

        self.frame_idx = 0
        self.recoveryCounters: defaultdict[str, int] = defaultdict(int)
        self.webcam = None

        self.objectTracker = None
        self.yolo26 = None
        self.pose_finder = None
        self.yolo_backend = "tensorrt" if self.model.endswith(".engine") else "torch"
        self.pose_backend = 'onnxruntime'
        self.pose_full_frame_fallback = False

        self.yolo_missing_reinit_attempts = 0
        self.last_yolo_reinit_time = 0.0
        self.yolo_reinit_backoff_seconds = 2.0
        self.max_yolo_reinit_attempts = 10

        # Store movement history.
        self.objectTrackHistory = defaultdict(lambda: [[], [], []])

        # Lock each tracked ID to one color for the full session.
        self.objectColorLock: dict[int, tuple[int, int, int]] = {}

        self.consecutiveCaptureFailures = 0
        self.maxConsecutiveCaptureFailures = 10
        self.maxWebcamResetAttempts = 3
        self.webcamResetAttempts = 0
        self.trackLastSeenFrame: dict[int, int] = {}

    def configure_webcam_latency(self) -> None:
        # Keep latency bounded when supported by the backend.
        if self.webcam is None:
            return
        try:
            self.webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def create_pose_finder(self, device: str) -> RTMPose:
        return RTMPose(
            onnx_model="end2end.onnx",
            model_input_size=(288, 384),
            to_openpose=self.openposeKpts,
            backend=self.pose_backend,
            device=device,
        )

    def create_object_tracker(self, device: str) -> StrongSort:
        use_half = device != "cpu"
        return StrongSort(
            reid_weights='clip_market1501.pt',
            device=device,
            half=use_half,
        )

    def switch_to_cpu(
        self,
        error_type: str,
        *,
        move_yolo: bool = False,
        recreate_pose: bool = False,
        **details,
    ) -> None:
        if not self.use_cuda:
            return

        self.recordRecovery(error_type, self.frame_idx, **details)
        self.use_cuda = False
        self.DEVICE_boxmot = "cpu"

        try:
            self.objectTracker = self.create_object_tracker("cpu")
        except Exception as tracker_error:
            self.failFast(
                "startup_tracker_cpu_fallback_failed",
                "Failed to reinitialize StrongSort tracker on CPU fallback.",
                error=str(tracker_error),
            )

        if move_yolo and self.yolo26 is not None:
            try:
                if self.yolo_backend == "tensorrt":
                    if not os.path.exists(self.pt_model):
                        self.failFast(
                            "startup_yolo_cpu_fallback_missing_pt",
                            "TensorRT engine cannot be moved to CPU and the PyTorch model is missing.",
                            engine_model=self.model,
                            pt_model=self.pt_model,
                        )
                    self.yolo26 = YOLO(self.pt_model, task="detect")
                    self.model = self.pt_model
                    self.yolo_backend = "torch"
                else:
                    self.yolo26.to('cpu')
            except Exception as yolo_error:
                self.failFast(
                    "startup_yolo_cpu_fallback_failed",
                    "Failed to move YOLO model to CPU fallback.",
                    error=str(yolo_error),
                )

        if recreate_pose:
            try:
                self.pose_finder = self.create_pose_finder('cpu')
            except Exception as pose_error:
                self.failFast(
                    "startup_pose_cpu_fallback_failed",
                    "Failed to initialize pose tracker on CPU fallback.",
                    error=str(pose_error),
                )

    def recordRecovery(self, error_type: str, frame: int, **details) -> None:
        self.recoveryCounters[error_type] += 1
        count = self.recoveryCounters[error_type]
        # Reduce log spam for frequent recoveries while preserving useful signal.
        should_log = count <= 3 or (count & (count - 1) == 0)
        if should_log:
            payload = {
                "type": error_type,
                "frame": frame,
                "count": count,
                **details,
            }
            print(f"[RECOVERED] {json.dumps(payload, default=str)}")

    def failFast(self, error_type: str, message: str, **details) -> None:
        self.recordRecovery(error_type, self.frame_idx, **details)
        print(f"Fatal startup error: {message}", file=sys.stderr)
        print(f"[RECOVERY_SUMMARY] {json.dumps(dict(sorted(self.recoveryCounters.items())), default=str)}")
        raise SystemExit(1)

    @staticmethod
    def randomColor() -> tuple[int, int, int]:
        h = random.uniform(0.0, 1.0)  # hue (0.0 to 1.0)
        s = random.uniform(0.5, 1.0)  # saturation
        v = random.uniform(0.2, 0.5)  # brightness (LOW = dark)

        r, g, b = colorsys.hsv_to_rgb(h, s, v)  # Return values in the range [0, 1]

        # OpenCV uses BGR and we multiply by 255 to convert to the range [0, 255].
        return (int(b * 255), int(g * 255), int(r * 255))

    def initialize(self) -> None:
        try:
            self.objectTracker = self.create_object_tracker(self.DEVICE_boxmot)
        except Exception as error:
            if self.use_cuda:
                self.switch_to_cpu("startup_tracker_cuda_init_failed", error=str(error))
            else:
                self.failFast("startup_tracker_init_failed", "Failed to initialize StrongSort tracker.", error=str(error))

        try:
            self.yolo26 = YOLO(self.model, task="detect")
        except Exception as error:
            self.failFast("startup_yolo_load_failed", "Failed to load YOLO model.", error=str(error))

        if self.yolo_backend == "torch" and self.use_cuda:
            try:
                self.yolo26.to('cuda')
            except Exception as error:
                self.switch_to_cpu("startup_yolo_cuda_move_failed", move_yolo=True, error=str(error))

        try:
            self.pose_finder = self.create_pose_finder("cuda" if self.use_cuda else "cpu")
        except Exception as error:
            if self.use_cuda:
                self.switch_to_cpu("startup_pose_cuda_init_failed", move_yolo=True, recreate_pose=True, error=str(error))
            else:
                self.failFast("startup_pose_init_failed", "Failed to initialize pose tracker.", error=str(error))

        self.webcam = cv2.VideoCapture(0)
        if not self.webcam.isOpened():
            self.webcam.release()
            self.failFast("startup_webcam_open_failed", "Unable to open webcam index 0.")

        self.configure_webcam_latency()

    def try_reinitialize_yolo(self) -> bool:
        if self.yolo_missing_reinit_attempts >= self.max_yolo_reinit_attempts:
            self.failFast(
                "runtime_yolo_reinit_attempts_exhausted",
                "YOLO model could not be recovered after maximum retry attempts.",
                attempts=self.yolo_missing_reinit_attempts,
                max_attempts=self.max_yolo_reinit_attempts,
            )

        try:
            self.yolo26 = YOLO(self.model, task="detect")

            if self.yolo_backend == "torch" and self.use_cuda:
                try:
                    self.yolo26.to('cuda')
                except Exception as error:
                    self.switch_to_cpu("runtime_yolo_cuda_move_failed", move_yolo=True, recreate_pose=True, error=str(error))

            self.yolo_missing_reinit_attempts = 0
            self.last_yolo_reinit_time = time.time()
            self.recordRecovery(
                "runtime_yolo_reinitialized",
                self.frame_idx,
                backend=self.yolo_backend,
                model=self.model,
            )
            return True
        except Exception as error:
            self.yolo_missing_reinit_attempts += 1
            self.last_yolo_reinit_time = time.time()
            self.recordRecovery(
                "runtime_yolo_reinit_failed",
                self.frame_idx,
                attempt=self.yolo_missing_reinit_attempts,
                error=str(error),
            )
            return False

    def process_loop(self) -> None:
        while self.webcam.isOpened() and self.captureFrames:
            ret, frame = self.webcam.read()
            if not ret:
                self.consecutiveCaptureFailures += 1
                self.recordRecovery(
                    "webcam_frame_read_failed",
                    self.frame_idx,
                    consecutive_failures=self.consecutiveCaptureFailures,
                )
                if self.consecutiveCaptureFailures >= self.maxConsecutiveCaptureFailures:
                    # Try to reset the webcam before giving up
                    if self.webcamResetAttempts < self.maxWebcamResetAttempts:
                        self.webcamResetAttempts += 1
                        current_reset_attempt = self.webcamResetAttempts
                        print(f"Attempting webcam reset (attempt {current_reset_attempt}/{self.maxWebcamResetAttempts})...")
                        try:
                            self.webcam.release()
                            time.sleep(0.5)  # Brief pause to allow hardware to settle
                            self.webcam = cv2.VideoCapture(0)
                            if self.webcam.isOpened():
                                self.configure_webcam_latency()
                                self.consecutiveCaptureFailures = 0
                                self.webcamResetAttempts = 0
                                self.recordRecovery("webcam_reset_success", self.frame_idx, reset_attempt=current_reset_attempt)
                                continue
                            else:
                                self.recordRecovery("webcam_reset_failed_reopen", self.frame_idx, reset_attempt=current_reset_attempt)
                        except Exception as reset_error:
                            self.recordRecovery("webcam_reset_exception", self.frame_idx, error=str(reset_error), reset_attempt=current_reset_attempt)
                    print("Failed to capture frame repeatedly and reset attempts exhausted. Exiting...")
                    self.captureFrames = False
                    break
                continue

            self.consecutiveCaptureFailures = 0

            # Use tracking and object detection.
            # If CUDA/cuDNN is mismatched at runtime, fallback to CPU once and continue.
            if self.yolo26 is None:
                self.recordRecovery("yolo_model_missing", self.frame_idx)
                now = time.time()
                retry_delay = min(30.0, self.yolo_reinit_backoff_seconds * (2 ** min(self.yolo_missing_reinit_attempts, 4)))
                if now - self.last_yolo_reinit_time >= retry_delay:
                    self.try_reinitialize_yolo()
                else:
                    # Prevent tight-loop CPU burn while waiting for next recovery attempt.
                    time.sleep(0.05)
                continue

            try:
                objectResults = self.yolo26(frame, verbose=False)
            except RuntimeError as error:
                error_text = str(error)
                is_cudnn_mismatch = (
                    "CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH" in error_text
                    or "CUDNN_BACKEND_TENSOR_DESCRIPTOR" in error_text
                )
                if self.use_cuda and is_cudnn_mismatch:
                    print("CUDA/cuDNN mismatch detected. Switching YOLO and pose tracking to CPU.")
                    self.switch_to_cpu(
                        "cuda_cudnn_fallback_to_cpu",
                        move_yolo=True,
                        recreate_pose=True,
                        device_before="cuda",
                        device_after="cpu",
                    )
                    try:
                        objectResults = self.yolo26(frame, device='cpu', verbose=False)
                    except Exception as fallback_error:
                        self.recordRecovery("yolo_cpu_fallback_exception", self.frame_idx, error=str(fallback_error))
                        continue
                else:
                    self.recordRecovery("yolo_runtime_exception", self.frame_idx, error=error_text)
                    continue
            except Exception as error:
                self.recordRecovery("yolo_inference_exception", self.frame_idx, error=str(error))
                continue

            img_show = frame.copy()
            det_results = []

            # Keep frame mutation in a single thread. OpenCV drawing is not thread-safe.
            try:
                det_results = processYOLOResults(
                    objectResults,
                    self.objectTracker,
                    self.objectTrackHistory,
                    img_show,
                    self.yolo26,
                    self.objectColorLock,
                    self.randomColor,
                    frame_idx=self.frame_idx,
                    errorCounters=self.recoveryCounters,
                    warn=lambda error_type, details: self.recordRecovery(error_type, self.frame_idx, **details),
                    track_last_seen_frame=self.trackLastSeenFrame,
                )
            except Exception as error:
                self.recordRecovery("yolo_processing_exception", self.frame_idx, error=str(error))

            try:
                should_run_pose = bool(det_results) or self.pose_full_frame_fallback
                if should_run_pose:
                    img_show = processPoseResults(
                        frame,
                        det_results,  # pose_inputs
                        img_show,
                        self.pose_finder,
                        self.openposeKpts,
                        errorCounters=self.recoveryCounters,
                        warn=lambda error_type, details: self.recordRecovery(error_type, self.frame_idx, **details),
                    )
            except Exception as error:
                self.recordRecovery("pose_processing_exception", self.frame_idx, error=str(error))

            # Draw results.
            try:
                cv2.imshow("Sieve AI", img_show)
            except cv2.error as error:
                self.recordRecovery("display_exception", self.frame_idx, error=str(error))
                continue

            # Check for exit.
            try:
                cv2.waitKey(1)
                if cv2.getWindowProperty("Sieve AI", cv2.WND_PROP_VISIBLE) < 1:
                    print("Exiting...")
                    self.captureFrames = False
                    break
            except cv2.error as error:
                self.recordRecovery("window_state_exception", self.frame_idx, error=str(error))
                continue

            # Increment frame counter only after successful processing.
            self.frame_idx += 1

    def cleanup(self) -> None:
        if self.recoveryCounters:
            print(f"[RECOVERY_SUMMARY] {json.dumps(dict(sorted(self.recoveryCounters.items())), default=str)}")
        self.trackLastSeenFrame.clear()
        if self.webcam is not None:
            self.webcam.release()
        cv2.destroyAllWindows()

    def run(self) -> None:
        try:
            self.initialize()
            self.process_loop()
        except KeyboardInterrupt:
            print("Interrupted by user. Exiting...")
        except Exception as error:
            self.recordRecovery("fatal_unhandled_exception", self.frame_idx, error=str(error))
            print(f"Fatal error: {error}", file=sys.stderr)
        finally:
            self.cleanup()


def RunSieveAI() -> None:
    sieve = SieveAI()
    sieve.run()


if __name__ == "__main__":
    RunSieveAI()