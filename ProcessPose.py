import numpy as np
from typing import Callable, Any
from rtmlib import RTMPose, draw_skeleton

def processPoseResults(
    frame: Any | None,
    pose_inputs: list,
    img_show: Any | None,
    pose_tracker: RTMPose,
    openPose: bool,
    errorCounters: dict[str, int] | None = None,
    warn: Callable[[str, dict], None] | None = None,
) -> Any:
    def recordRecovery(error_type: str, **details) -> None:
        if errorCounters is not None and warn is None:
            errorCounters[error_type] = errorCounters.get(error_type, 0) + 1
        if warn is not None:
            warn(error_type, details)

    if frame is None or img_show is None:
        recordRecovery("pose_invalid_frame_or_image")
        return img_show

    # Input specific bboxes but fallback to full frame pose detection if bbox errors occur
    bboxes: list[list[float]] = []
    frame_h, frame_w = frame.shape[:2]
    if pose_inputs:
        for bbox in pose_inputs:
            if bbox is None or len(bbox) < 4:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            except (TypeError, ValueError):
                continue

            x1 = max(0, min(frame_w - 1, x1))
            y1 = max(0, min(frame_h - 1, y1))
            x2 = max(0, min(frame_w - 1, x2))
            y2 = max(0, min(frame_h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            bboxes.append([x1, y1, x2, y2])

    try:
        pose_out = pose_tracker(image=frame, bboxes=bboxes)
    except Exception as error:
        recordRecovery("pose_tracker_exception", error=str(error))
        return img_show
    
    if pose_out is None or len(pose_out) < 2:
        return img_show
    
    keypoints, scores = pose_out[:2]

    keypoints = np.asarray(keypoints)
    scores = np.asarray(scores)
    if keypoints.ndim < 2 or scores.ndim < 1:
        recordRecovery("pose_invalid_output_shape", keypoints_ndim=int(keypoints.ndim), scores_ndim=int(scores.ndim))
        return img_show

    if keypoints.size == 0 or scores.size == 0:
        return img_show

    try:
        img_show = draw_skeleton(
            img_show,
            keypoints,
            scores,
            openpose_skeleton=openPose,
            kpt_thr=0.43,
        )
    except Exception as error:
        recordRecovery("pose_draw_exception", error=str(error))
        return img_show
    
    return img_show
