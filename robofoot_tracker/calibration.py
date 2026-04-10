"""Homography calibration for pixel-to-field coordinate mapping.

Corner order convention: TL, TR, BR, BL (top-left, top-right, bottom-right, bottom-left).
"""

from __future__ import annotations

import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

from .models import CalibrationData, FieldDimensions


def calibrate_from_points(
    pixel_points: list[tuple[float, float]],
    field_dims: FieldDimensions,
) -> CalibrationData:
    """Compute homography from 4 pixel corners to field coordinates.

    Args:
        pixel_points: 4 pixel-space points in order TL, TR, BR, BL.
        field_dims: Physical field dimensions.

    Returns:
        CalibrationData with the computed homography matrix.
    """
    if len(pixel_points) != 4:
        raise ValueError(f"Expected 4 points, got {len(pixel_points)}")

    w, h = field_dims.width_cm, field_dims.height_cm
    src = np.array(pixel_points, dtype=np.float32)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    matrix, _ = cv2.findHomography(src, dst)
    if matrix is None:
        raise RuntimeError("Homography computation failed")

    return CalibrationData(homography_matrix=matrix, src_points=src, dst_points=dst)


def estimate_distortion(
    pixel_points_8: list[tuple[float, float]],
    field_dims: FieldDimensions,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate camera matrix and distortion coefficients from 8 coplanar points.

    Args:
        pixel_points_8: 8 pixel points in order TL, Top-Mid, TR, Right-Mid,
                        BR, Bottom-Mid, BL, Left-Mid.
        field_dims: Physical field dimensions.
        image_size: (height, width) of the image.

    Returns:
        (camera_matrix, dist_coeffs) from cv2.calibrateCamera.
    """
    if len(pixel_points_8) != 8:
        raise ValueError(f"Expected 8 points, got {len(pixel_points_8)}")

    w, h = field_dims.width_cm, field_dims.height_cm
    obj_pts = np.array([
        [0, 0, 0], [w / 2, 0, 0], [w, 0, 0], [w, h / 2, 0],
        [w, h, 0], [w / 2, h, 0], [0, h, 0], [0, h / 2, 0],
    ], dtype=np.float32)
    img_pts = np.array(pixel_points_8, dtype=np.float32).reshape(-1, 1, 2)

    h_img, w_img = image_size
    focal = max(h_img, w_img)
    cam_init = np.array([[focal, 0, w_img / 2],
                         [0, focal, h_img / 2],
                         [0, 0, 1]], dtype=np.float64)
    dist_init = np.zeros(5, dtype=np.float64)

    _, cam_mtx, dist, _, _ = cv2.calibrateCamera(
        [obj_pts], [img_pts], (w_img, h_img), cam_init, dist_init,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT,
    )
    flat = dist.ravel()
    k1, k2 = flat[0], flat[1]
    if abs(k1) > 2.0 or abs(k2) > 2.0:
        raise ValueError(f"Distortion coefficients out of range (k1={k1:.3f}, k2={k2:.3f}), discarding")
    return cam_mtx, dist


def calibrate_interactive(
    frame: np.ndarray,
    field_dims: FieldDimensions,
) -> CalibrationData:
    """Open an OpenCV window for the user to click 4 corners, then auto-generate midpoints.

    Flow:
    1. User clicks 4 corners: TL, TR, BR, BL.
    2. After the 4th click, 4 midpoints are auto-generated (Top-Mid, Right-Mid,
       Bottom-Mid, Left-Mid).
    3. All 8 points are displayed with Bézier curves. User can drag any point.
    4. Enter/Space confirms; ESC cancels; right-click undoes (removes all 4
       midpoints at once back to 4 corners, then individual corner undo).

    With 8 points: estimates distortion then computes homography from corners.
    With 4-7 points (early confirm): computes homography from corners (indices 0,2,4,6).
    Fewer than 4 on confirm raises RuntimeError.
    """
    points: list[tuple[float, float]] = []
    corner_labels = ["TL", "TR", "BR", "BL"]
    mid_labels = ["Top-Mid", "Right-Mid", "Bottom-Mid", "Left-Mid"]
    labels = corner_labels + mid_labels
    dragging_idx: list[int] = [-1]  # mutable container for closure
    midpoints_generated: list[bool] = [False]

    def _auto_midpoints() -> None:
        """Compute and append 4 midpoints from the 4 corners."""
        tl, tr, br, bl = points[0], points[1], points[2], points[3]
        points.append(((tl[0] + tr[0]) / 2, (tl[1] + tr[1]) / 2))  # Top-Mid
        points.append(((tr[0] + br[0]) / 2, (tr[1] + br[1]) / 2))  # Right-Mid
        points.append(((br[0] + bl[0]) / 2, (br[1] + bl[1]) / 2))  # Bottom-Mid
        points.append(((bl[0] + tl[0]) / 2, (bl[1] + tl[1]) / 2))  # Left-Mid
        midpoints_generated[0] = True

    def _find_near(x: int, y: int, threshold: float = 15.0) -> int:
        for i, pt in enumerate(points):
            if ((pt[0] - x) ** 2 + (pt[1] - y) ** 2) ** 0.5 < threshold:
                return i
        return -1

    def _on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            near = _find_near(x, y)
            if near >= 0:
                dragging_idx[0] = near
            elif len(points) < 4 and not midpoints_generated[0]:
                points.append((float(x), float(y)))
                if len(points) == 4:
                    _auto_midpoints()
        elif event == cv2.EVENT_MOUSEMOVE and dragging_idx[0] >= 0:
            points[dragging_idx[0]] = (float(x), float(y))
        elif event == cv2.EVENT_LBUTTONUP:
            dragging_idx[0] = -1
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            if midpoints_generated[0]:
                del points[4:]
                midpoints_generated[0] = False
            else:
                points.pop()

    window = "Calibration - click 4 corners"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    max_w, max_h = 1280, 720
    scale = min(max_w / frame.shape[1], max_h / frame.shape[0])
    cv2.resizeWindow(window, int(frame.shape[1] * scale), int(frame.shape[0] * scale))
    cv2.setMouseCallback(window, _on_mouse)

    confirmed = False
    while True:
        display = frame.copy()
        for i, pt in enumerate(points):
            cv2.circle(display, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
            cv2.putText(display, labels[i], (int(pt[0]) + 8, int(pt[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if len(points) >= 2:
            if len(points) == 8:
                from .geometry import bezier_curve_points as _bezier_curve_points
                pts_arr = np.array(points, dtype=np.float32)
                edges = [(0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0)]
                for i0, i1, i2 in edges:
                    bpts = _bezier_curve_points(pts_arr[i0], pts_arr[i1], pts_arr[i2])
                    cv2.polylines(display, [bpts.reshape(-1, 1, 2)], isClosed=False, color=(0, 255, 0), thickness=1)
            else:
                pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(display, [pts], isClosed=False, color=(0, 255, 0), thickness=1)
        if midpoints_generated[0]:
            instr = "8/8 points | Drag to adjust | Enter:confirm | RClick:undo | ESC:cancel"
        else:
            label = corner_labels[len(points)] if len(points) < 4 else "?"
            instr = f"Click corner {len(points)+1}/4: {label} | RClick:undo | ESC:cancel"
        cv2.putText(display, instr, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(window, display)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break
        if key in (13, 32):  # Enter or Space
            confirmed = True
            break

    cv2.destroyWindow(window)

    if not confirmed:
        raise RuntimeError("Calibration aborted by user")
    if len(points) < 4:
        raise RuntimeError(f"Need at least 4 points, got {len(points)}")

    corners = list(points[:4])
    cal = calibrate_from_points(corners, field_dims)

    if len(points) == 8:
        # Reorder to TL, Top-Mid, TR, Right-Mid, BR, Bottom-Mid, BL, Left-Mid
        # for estimate_distortion compatibility.
        ordered_8 = [points[0], points[4], points[1], points[5],
                      points[2], points[6], points[3], points[7]]
        all_8_arr = np.array(ordered_8, dtype=np.float32).reshape(-1, 1, 2)
        try:
            cam_mtx, dist = estimate_distortion(ordered_8, field_dims, frame.shape[:2])
            corners_arr = np.array(corners, dtype=np.float32).reshape(-1, 1, 2)
            undistorted = cv2.undistortPoints(corners_arr, cam_mtx, dist, P=cam_mtx)
            corners_undist = [tuple(pt[0]) for pt in undistorted]
            cal = calibrate_from_points(corners_undist, field_dims)
            cal.camera_matrix = cam_mtx
            cal.dist_coeffs = dist
            all_undist = cv2.undistortPoints(all_8_arr, cam_mtx, dist, P=cam_mtx)
            cal.all_points = all_undist.reshape(-1, 2).astype(np.float32)
        except Exception:
            logger.warning("Distortion estimation failed; falling back to 4-point calibration")
            cal.all_points = all_8_arr.reshape(-1, 2).astype(np.float32)

    return cal


def _compute_hsv_ranges(
    samples: list[np.ndarray],
    h_pad: int = 15,
    s_pad: int = 50,
    v_pad: int = 50,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compute HSV lower/upper ranges from a list of individual HSV pixel values.

    Handles red hue wrapping (samples near both 0 and 180).
    """
    arr = np.array(samples, dtype=np.float32)
    h_vals = arr[:, 0]

    has_low = np.any(h_vals < 20)
    has_high = np.any(h_vals > 160)
    needs_wrap = has_low and has_high

    if needs_wrap:
        low_mask = h_vals < 90
        high_mask = ~low_mask
        ranges = []
        for mask in (low_mask, high_mask):
            if not np.any(mask):
                continue
            sub = arr[mask]
            lo = sub.min(axis=0)
            hi = sub.max(axis=0)
            lower = np.array([
                max(lo[0] - h_pad, 0),
                max(lo[1] - s_pad, 0),
                max(lo[2] - v_pad, 0),
            ], dtype=np.uint8)
            upper = np.array([
                min(hi[0] + h_pad, 180),
                min(hi[1] + s_pad, 255),
                min(hi[2] + v_pad, 255),
            ], dtype=np.uint8)
            ranges.append((lower, upper))
    else:
        lo = arr.min(axis=0)
        hi = arr.max(axis=0)
        lower = np.array([
            max(lo[0] - h_pad, 0),
            max(lo[1] - s_pad, 0),
            max(lo[2] - v_pad, 0),
        ], dtype=np.uint8)
        upper = np.array([
            min(hi[0] + h_pad, 180),
            min(hi[1] + s_pad, 255),
            min(hi[2] + v_pad, 255),
        ], dtype=np.uint8)
        ranges = [(lower, upper)]

    return ranges


def calibrate_colors_interactive(frame: np.ndarray) -> "ColorConfig":
    """Open an OpenCV window for the user to sample HSV colors for each tag color.

    Iterates 7 colors: blue, yellow, red, green, cyan, purple, orange.
    Left-click samples a 5x5 HSV patch. Right-click undoes last sample.
    Enter/Space confirms. S skips (empty ranges). ESC cancels all.

    Returns:
        ColorConfig with sampled HSV ranges (or defaults on ESC).
    """
    from .color_config import ColorConfig

    COLOR_ORDER = ["blue", "yellow", "red", "green", "cyan", "purple", "orange"]
    H_PAD, S_PAD, V_PAD = 15, 50, 50
    config = ColorConfig()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_img, w_img = frame.shape[:2]

    for color_name in COLOR_ORDER:
        samples: list[np.ndarray] = []
        click_points: list[tuple[int, int]] = []

        def _on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
            if event == cv2.EVENT_LBUTTONDOWN:
                x0, y0 = max(x - 2, 0), max(y - 2, 0)
                x1, y1 = min(x + 3, w_img), min(y + 3, h_img)
                patch = hsv_frame[y0:y1, x0:x1].reshape(-1, 3)
                samples.extend(patch.astype(np.float32))
                click_points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN and click_points:
                # Remove last click's pixels (up to 25 pixels from 5x5 patch)
                x_last, y_last = click_points.pop()
                x0, y0 = max(x_last - 2, 0), max(y_last - 2, 0)
                x1, y1 = min(x_last + 3, w_img), min(y_last + 3, h_img)
                n_pixels = (y1 - y0) * (x1 - x0)
                del samples[-n_pixels:]

        window = f"Color calibration - {color_name}"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        max_w, max_h = 1280, 720
        scale = min(max_w / frame.shape[1], max_h / frame.shape[0])
        cv2.resizeWindow(window, int(frame.shape[1] * scale), int(frame.shape[0] * scale))
        cv2.setMouseCallback(window, _on_mouse)

        confirmed = False
        skipped = False
        cancelled = False
        while True:
            display = frame.copy()
            # Draw click point circles
            for cp in click_points:
                cv2.circle(display, cp, 5, (0, 255, 0), -1)
            # Live mask preview
            if samples:
                ranges = _compute_hsv_ranges(samples, H_PAD, S_PAD, V_PAD)
                mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
                for lo, hi in ranges:
                    mask |= cv2.inRange(hsv_frame, lo, hi)
                overlay = display.copy()
                overlay[mask > 0] = (0, 255, 0)
                cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
            txt = [
                f"Color: {color_name} | Samples: {len(click_points)}",
                "LClick:sample | RClick:undo | Enter:confirm | S:skip | ESC:cancel",
            ]
            for i, line in enumerate(txt):
                cv2.putText(display, line, (10, 25 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(window, display)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                cancelled = True
                break
            if key in (13, 32):
                confirmed = True
                break
            if key in (ord("s"), ord("S")):
                skipped = True
                break

        cv2.destroyWindow(window)

        if cancelled:
            return ColorConfig()

        if skipped or not samples:
            config.set_range(color_name, [])
            continue

        ranges = _compute_hsv_ranges(samples, H_PAD, S_PAD, V_PAD)
        config.set_range(color_name, ranges)

    return config


def transform_point(
    px_point: tuple[float, float],
    homography: np.ndarray,
) -> tuple[float, float]:
    """Map a single pixel coordinate to field coordinates using *homography*.

    Returns:
        (x_cm, y_cm) in field coordinate space.
    """
    src = np.array([[[px_point[0], px_point[1]]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, homography)
    return (float(dst[0, 0, 0]), float(dst[0, 0, 1]))


