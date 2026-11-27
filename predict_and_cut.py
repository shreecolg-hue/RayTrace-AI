import math
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw
from glass_cut_optimizer import process_glass_laser_cut
from train_recovery_model import predict_recovery_yield
import cv2
import numpy as np

DISPLAY_LABELS = {
    "good_condition": "good condition",
    "electric_faulty": "electric faulty",
    "physical_damage": "physical damage"
}

DEFAULT_DEFECT_RATIO_BY_CLASS = {
    "good condition": 0.05,
    "electric faulty": 0.18,
    "physical damage": 0.35,
}


def crop_solar_panel(image_path, cropped_output_path="cropped_panel.jpg", padding_px=8):
    """
    Automatically detects the largest rectangular solar panel contour in the image,
    applies perspective rectification (if tilted) or tight bounding-box crop,
    and saves the isolated panel image.
    """
    img = cv2.imread(image_path)
    if img is None:
        return image_path

    orig = img.copy()
    h, w = img.shape[:2]
    total_area = h * w

    # 1. Preprocess: grayscale -> blur -> Canny edge detection + morphological closing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 2. Find external contours and sort largest-first
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    panel_contour = None
    for cnt in contours[:10]:
        area = cv2.contourArea(cnt)
        # Ignore tiny noise (< 10% of image) or border artifacts (> 98% of image)
        if area < 0.10 * total_area or area > 0.98 * total_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # Prefer a 4-corner quadrilateral (the solar panel frame)
        if len(approx) == 4:
            panel_contour = approx
            break

        # Fallback: take the largest valid contour's bounding box
        if panel_contour is None:
            panel_contour = cnt

    # If no clear panel contour is found, keep original image
    if panel_contour is None:
        return image_path

    # 3A. If 4 corners detected -> Perspective Warp (straightens tilted panels cleanly!)
    if len(panel_contour) == 4:
        pts = panel_contour.reshape(4, 2).astype("float32")
        # Order points: top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr, br, bl) = rect
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_w = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_h = max(int(height_a), int(height_b))

        if max_w > 50 and max_h > 50:
            dst = np.array([
                [0, 0],
                [max_w - 1, 0],
                [max_w - 1, max_h - 1],
                [0, max_h - 1]
            ], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(orig, M, (max_w, max_h))
            cv2.imwrite(cropped_output_path, warped)
            return cropped_output_path

    # 3B. Fallback: Tight Axis-Aligned Bounding Box Crop
    x, y, bw, bh = cv2.boundingRect(panel_contour)
    x1 = max(0, x - padding_px)
    y1 = max(0, y - padding_px)
    x2 = min(w, x + bw + padding_px)
    y2 = min(h, y + bh + padding_px)

    cropped = orig[y1:y2, x1:x2]
    cv2.imwrite(cropped_output_path, cropped)
    return cropped_output_path

def _summarize_trajectory(trajectory_coords):
    """Derive cut segment count and total cut length (in coordinate units/mm) from trajectory points."""
    if not trajectory_coords:
        return 0, 0.0

    total_length = 0.0
    num_segments = 0

    if isinstance(trajectory_coords[0], (list, tuple)) and len(trajectory_coords) > 0:
        first_elem = trajectory_coords[0]
        # Case 1: flat list of (x, y) coordinates -> [(x1, y1), (x2, y2), ...]
        if len(first_elem) == 2 and isinstance(first_elem[0], (int, float)):
            num_segments = max(1, len(trajectory_coords) - 1)
            for i in range(1, len(trajectory_coords)):
                x1, y1 = trajectory_coords[i - 1]
                x2, y2 = trajectory_coords[i]
                total_length += math.hypot(x2 - x1, y2 - y1)
        # Case 2: list of segments/paths -> [[(x1, y1), (x2, y2)], ...]
        else:
            for seg in trajectory_coords:
                if len(seg) >= 2:
                    num_segments += len(seg) - 1
                    for i in range(1, len(seg)):
                        x1, y1 = seg[i - 1]
                        x2, y2 = seg[i]
                        total_length += math.hypot(x2 - x1, y2 - y1)
            num_segments = max(1, num_segments)

    return num_segments, round(total_length, 2)


def _compute_glass_only_perimeter(image_path, output_path="classified_laser_output.jpg", margin_ratio=0.04):
    """
    Generate an outer perimeter glass-retention trajectory that releases the intact
    front tempered glass sheet while leaving electrically damaged semiconductor cells untouched.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)

    # Closed rectangular perimeter waypoints for full glass sheet retention
    perimeter_waypoints = [
        (mx, my),
        (w - mx, my),
        (w - mx, h - my),
        (mx, h - my),
        (mx, my),
    ]

    draw = ImageDraw.Draw(img)
    # Draw yellow glass-retention perimeter cut
    draw.line(perimeter_waypoints, fill=(250, 204, 21), width=4)

    # Shade interior semiconductor region to indicate exclusion from laser recovery
    inner_mx = int(w * 0.12)
    inner_my = int(h * 0.12)
    draw.rectangle(
        [(inner_mx, inner_my), (w - inner_mx, h - inner_my)],
        outline=(245, 158, 11),
        width=2
    )
    draw.text(
        (inner_mx + 10, inner_my + 10),
        "GLASS PERIMETER CUT ONLY | DAMAGED SEMICONDUCTORS EXCLUDED",
        fill=(250, 204, 21),
    )

    img.save(output_path)
    return output_path, perimeter_waypoints


def _render_thermal_crushing_plan(image_path, output_path="classified_laser_output.jpg"):
    """
    Skip laser trajectory for physically damaged modules and mark image for
    standard thermal delamination and mechanical glass crushing.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    banner_height = 46
    draw.rectangle([(0, 0), (w, banner_height)], fill=(153, 27, 27))
    draw.text(
        (14, 15),
        "LASER BYPASSED: ROUTE TO THERMAL HEATING & GLASS CRUSHING",
        fill=(255, 255, 255),
    )

    img.save(output_path)
    return output_path, []


def load_trained_model(checkpoint_path="best_solar_classifier.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]

    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, len(class_names))
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, class_names, device


def analyze_solar_panel(
    image_path,
    model,
    class_names,
    device,
    defect_area_ratio=None,
    laser_power_w=45.0,
    recovery_model_path="recovery_yield_model.joblib",
):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        conf, pred_idx = torch.max(probs, 0)

    raw_class = class_names[pred_idx.item()]
    predicted_label = DISPLAY_LABELS.get(raw_class, raw_class)
    confidence = round(conf.item() * 100, 2)

    output_file = "classified_laser_output.jpg"

    # --- THREE-TIER RECOVERY OPTIMIZATION ROUTING ---
    if predicted_label == "good condition":
        # 1. CLEAN: Full laser path to retain all glass, silicon, and metals
        output_img_path, trajectory_coords = process_glass_laser_cut(
            image_path, output_path=output_file
        )
        effective_laser_power = float(laser_power_w)
        recovery_strategy = "FULL RETENTION: Glass, Silicon & Metals Preserved"
        process_steps = [
            "1. Execute full perimeter and internal cell-matrix laser separation.",
            "2. Delaminate intact front tempered glass sheet without thermal stress.",
            "3. Extract undamaged silicon wafers and silver/copper busbars for direct reuse."
        ]

    elif predicted_label == "electric faulty":
        # 2. ELECTRICALLY DEFECTED: Perimeter glass cut only; leave damaged semiconductors off
        output_img_path, trajectory_coords = _compute_glass_only_perimeter(
            image_path, output_path=output_file
        )
        effective_laser_power = float(laser_power_w)
        recovery_strategy = "SELECTIVE GLASS RETENTION: Intact Glass Saved | Damaged Semiconductors Excluded"
        process_steps = [
            "1. Execute outer perimeter laser delamination to release the intact front glass sheet.",
            "2. Bypass interior electrically damaged semiconductor cells to prevent cross-contamination.",
            "3. Reclaim intact glass for module remanufacturing; route isolated cells to chemical metal refining."
        ]

    else:
        # 3. PHYSICALLY DAMAGED: Skip laser cutting; follow normal thermal heating + crushing steps
        output_img_path, trajectory_coords = _render_thermal_crushing_plan(
            image_path, output_path=output_file
        )
        effective_laser_power = 0.0
        recovery_strategy = "STANDARD RECYCLING: Thermal Heating & Mechanical Glass Crushing"
        process_steps = [
            "1. Bypass laser cutting (shattered/compromised glass cannot be reclaimed intact).",
            "2. Heat panel in thermal delamination furnace (~450-500 C) to decompose EVA encapsulant.",
            "3. Mechanically crush and screen separated glass into cullet, followed by silicon and metal sorting."
        ]

    num_cut_segments, total_cut_length_mm = _summarize_trajectory(trajectory_coords)
    effective_defect_ratio = (
        defect_area_ratio
        if defect_area_ratio is not None
        else DEFAULT_DEFECT_RATIO_BY_CLASS.get(predicted_label, 0.15)
    )

    predicted_yield_pct = predict_recovery_yield(
        panel_class=predicted_label,
        class_confidence=conf.item(),
        defect_area_ratio=effective_defect_ratio,
        num_cut_segments=num_cut_segments,
        total_cut_length_mm=total_cut_length_mm,
        laser_power_w=effective_laser_power,
        model_path=recovery_model_path,
    )

    return {
        "classification": predicted_label,
        "confidence_percent": confidence,
        "recovery_strategy": recovery_strategy,
        "process_steps": process_steps,
        "output_image": output_img_path,
        "laser_trajectory_points": trajectory_coords,
        "predicted_recovery_yield_pct": predicted_yield_pct,
        "cut_summary": {
            "num_cut_segments": num_cut_segments,
            "total_cut_length_mm": total_cut_length_mm,
            "defect_area_ratio": round(float(effective_defect_ratio), 4),
            "laser_power_w": effective_laser_power,
        },
    }