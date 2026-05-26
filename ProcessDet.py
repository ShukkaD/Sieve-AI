import cv2
from ultralytics import YOLO
from collections import defaultdict
from typing import Callable, Any
from ultralytics.engine.results import Results

import numpy as np


def _resolve_class_name(modelType: YOLO, cls: int) -> str:
    names = getattr(modelType, "names", {})
    if isinstance(names, dict):
        return str(names.get(cls, cls))
    if isinstance(names, list) and 0 <= cls < len(names):
        return str(names[cls])
    return str(cls)

def generateTrackerInputs(
    results: list[Results],
    frame,
) -> np.ndarray:

    if not results or not hasattr(results[0], "boxes") or results[0].boxes is None:
        return np.empty((0, 6), dtype=np.float32)

    dets = results[0].boxes.data.cpu().numpy()
    dets = np.asarray(dets, dtype=np.float32)

    if dets.ndim != 2 or dets.shape[1] < 6:
        return np.empty((0, 6), dtype=np.float32)

    # BoxMOT expects [x1, y1, x2, y2, conf, cls]
    dets = dets[:, :6]

    # Keep only finite rows to avoid numerical issues in Kalman update
    finite_mask = np.isfinite(dets).all(axis=1)
    dets = dets[finite_mask]

    if dets.size == 0:
        return np.empty((0, 6), dtype=np.float32)

    if frame is not None:
        frame_h, frame_w = frame.shape[:2]
        
        # Before clipping, validate coordinate order to catch inverted boxes from YOLO
        valid_coords_mask = (dets[:, 0] < dets[:, 2]) & (dets[:, 1] < dets[:, 3])
        dets = dets[valid_coords_mask]
        
        if dets.size == 0:
            return np.empty((0, 6), dtype=np.float32)
        
        dets[:, 0] = np.clip(dets[:, 0], 0, frame_w - 1)
        dets[:, 1] = np.clip(dets[:, 1], 0, frame_h - 1)
        dets[:, 2] = np.clip(dets[:, 2], 0, frame_w - 1)
        dets[:, 3] = np.clip(dets[:, 3], 0, frame_h - 1)

    # Require positive-area boxes, reasonable minimum size, and high-confidence detections.
    # Prune Low-confidence detections (< 0.3)
    # Prune Tiny boxes (< 10 pixels per side)
    width = dets[:, 2] - dets[:, 0]
    height = dets[:, 3] - dets[:, 1]
    confidence = dets[:, 4]
    valid_mask = (width >= 10.0) & (height >= 10.0) & (confidence <= 1.0)
    dets = dets[valid_mask]

    if dets.size == 0:
        return np.empty((0, 6), dtype=np.float32)

    return dets.astype(np.float32, copy=False)



MAX_STALE_FRAMES = 20  # Stop drawing history if not updated for 20 frames
MAX_DEAD_TRACK_RETENTION = 100  # Prune track memory every 100 frames to prevent unbounded growth.

def processYOLOResults(
    results: list[Results],
    tracker: Any,
    trackHistoryDict: defaultdict[list[list, list, list]],
    currentFrame: Any | None,
    modelType: YOLO,
    colorLockDict: dict[int, tuple[int, int, int]],
    colorGenerator: Callable[[], tuple[int, int, int]],
    frame_idx: int = 0,
    errorCounters: dict[str, int] | None = None,
    warn: Callable[[str, dict], None] | None = None,
    track_last_seen_frame: dict[int, int] | None = None,
):

    pose_inputs = []

    def recordRecovery(error_type: str, **details) -> None:
        if errorCounters is not None and warn is None:
            errorCounters[error_type] = errorCounters.get(error_type, 0) + 1
        if warn is not None:
            warn(error_type, details)

    if currentFrame is None or not hasattr(currentFrame, "shape") or currentFrame.ndim < 2:
        recordRecovery("yolo_invalid_frame_input")
        return pose_inputs

    frame_h, frame_w = currentFrame.shape[:2]
    if track_last_seen_frame is None:
        track_last_seen_frame = {}

    dets = generateTrackerInputs(results, currentFrame)
    if dets.size == 0:
        recordRecovery("tracker_empty_detections")
        return pose_inputs

    try:
        tracks = tracker.update(dets, currentFrame, embs=None)
    except np.linalg.LinAlgError as error:
        # Recover from occasional non-positive-definite covariance state in the tracker
        # Clear active tracks and tracks
        if hasattr(tracker, "active_tracks"):
            tracker.active_tracks = []
        if hasattr(tracker, "tracks"):
            tracker.tracks = []
        
        recordRecovery("tracker_linalg_recovery", error=str(error), num_dets=int(len(dets)))
        
        # Use track history to maintain visual continuity, but only for recent tracks
        for objectTrackId, track_history in trackHistoryDict.items():
            if len(track_history[0]) > 0:  # Check if position history exists and has entries
                # Only draw history if the track was seen recently (within MAX_STALE_FRAMES)
                last_seen = track_last_seen_frame.get(objectTrackId, frame_idx)
                if frame_idx - last_seen > MAX_STALE_FRAMES:
                    continue  # Skip stale tracks; allows tracker to recover with fresh detections
                
                x1, y1, x2, y2 = track_history[0][-1]  # Use most recent position
                x1 = max(0, min(frame_w - 1, x1))
                y1 = max(0, min(frame_h - 1, y1))
                x2 = max(0, min(frame_w - 1, x2))
                y2 = max(0, min(frame_h - 1, y2))
                
                if x2 > x1 and y2 > y1:  # Only draw if valid box
                    if objectTrackId not in colorLockDict:
                        colorLockDict[objectTrackId] = colorGenerator()
                    color = colorLockDict[objectTrackId]
                    try:
                        cv2.rectangle(currentFrame, (x1, y1), (x2, y2), color, 2)
                    except cv2.error as error:
                        recordRecovery("tracker_recovery_draw_exception", error=str(error))
        return pose_inputs
    except Exception as error:
        # Skip the current frame for any unexpected tracker errors
        recordRecovery("tracker_update_exception", error=str(error), num_dets=int(len(dets)))
        return pose_inputs

    if tracks is None or len(tracks) == 0:
        # Periodically prune dead tracks to prevent unbounded memory growth
        if frame_idx > 0 and frame_idx % MAX_DEAD_TRACK_RETENTION == 0:
            dead_ids = set(track_last_seen_frame.keys())
            for dead_id in dead_ids:
                if dead_id in track_last_seen_frame and frame_idx - track_last_seen_frame[dead_id] > MAX_STALE_FRAMES:
                    del track_last_seen_frame[dead_id]
                    trackHistoryDict.pop(dead_id, None)
                    colorLockDict.pop(dead_id, None)
        return pose_inputs

    # Track active IDs in this frame for memory cleanup
    current_active_ids = set()
    for track in tracks:
        if len(track) < 8:
            recordRecovery("tracker_invalid_track_row", reason="short_row", row_len=int(len(track)))
            continue

        if not np.isfinite(track[:8]).all():
            recordRecovery("tracker_invalid_track_row", reason="non_finite")
            continue

        # StrongSort output: [x1, y1, x2, y2, track_id, conf, cls, det_ind]
        x1, y1, x2, y2 = tuple(map(lambda x: int(np.round(x)), track[:4]))
        objectTrackId = int(track[4])
        current_active_ids.add(objectTrackId)
        conf = float(track[5])
        cls = int(track[6]) 

        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(0, min(frame_w - 1, x2))
        y2 = max(0, min(frame_h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            recordRecovery("tracker_invalid_track_row", reason="non_positive_area")
            continue

        # Save history and update the frame number for this track
        trackHistoryDict[objectTrackId][0].append((x1, y1, x2, y2))
        track_last_seen_frame[objectTrackId] = frame_idx


        if objectTrackId not in colorLockDict:
            colorLockDict[objectTrackId] = colorGenerator()

        color = colorLockDict[objectTrackId]

        # Create and draw the bounding box to the frame
        try:
            cv2.rectangle(currentFrame, (x1, y1), (x2, y2), color, 2)
        except cv2.error as error:
            recordRecovery("yolo_draw_box_exception", error=str(error))
            continue

        # Draw the class name and track ID on the frame with a background for better visibility
        class_name = _resolve_class_name(modelType, cls)

        if class_name == "person":
            pose_inputs.append([x1, y1, x2, y2])

        label = f"{class_name.capitalize()} {conf * 100:.2f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_thickness = 1
        pad_x = 4
        pad_y = 4

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

        # Prefer above the box; if there isn't room, place it below
        label_h = text_h + baseline + (pad_y * 2)
        if y1 - label_h >= 0:
            bg_top = y1 - label_h
            bg_bottom = y1
        else:
            bg_top = y1
            bg_bottom = min(frame_h - 1, y1 + label_h)

        # Expand the background to the full label width and shift left if needed to keep text on-screen.
        bg_left = x1
        bg_right = x1 + text_w + (pad_x * 2)
        if bg_right > frame_w - 1:
            overflow = bg_right - (frame_w - 1)
            bg_left = max(0, bg_left - overflow)
            bg_right = frame_w - 1

        try:
            cv2.rectangle(currentFrame, (bg_left, bg_top), (bg_right, bg_bottom), color, -1)
        except cv2.error as error:
            recordRecovery("yolo_draw_label_bg_exception", error=str(error))
            continue

        text_x = bg_left + pad_x
        text_y = bg_top + pad_y + text_h
        try:
            cv2.putText(currentFrame, label, (text_x, text_y), font, font_scale, (255, 255, 255), text_thickness)
        except cv2.error as error:
            recordRecovery("yolo_draw_text_exception", error=str(error))
            continue

        # Keep last 15 points
        if len(trackHistoryDict[objectTrackId][0]) > 15:
            try:
                trackHistoryDict[objectTrackId][0].pop(0)
            except IndexError:
                pass
    
    # Periodically prune dead tracks to prevent unbounded memory growth
    if frame_idx > 0 and frame_idx % MAX_DEAD_TRACK_RETENTION == 0:
        dead_ids = set(track_last_seen_frame.keys()) - current_active_ids
        for dead_id in dead_ids:
            if dead_id in track_last_seen_frame and frame_idx - track_last_seen_frame[dead_id] > MAX_STALE_FRAMES:
                del track_last_seen_frame[dead_id]
                trackHistoryDict.pop(dead_id, None)
                colorLockDict.pop(dead_id, None)

    return pose_inputs
