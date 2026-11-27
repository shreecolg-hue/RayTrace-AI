import cv2
import numpy as np

def find_max_rectangle(matrix):
    """
    Finds the largest rectangle of 1s in a binary matrix (0=defect, 1=clean glass).
    Returns (max_area, top_left_coord, bottom_right_coord) -> (area, (x1, y1), (x2, y2))
    """
    rows = len(matrix)
    if rows == 0:
        return 0, (0, 0), (0, 0)
    cols = len(matrix[0])

    heights = [0] * cols
    max_area = 0
    best_coords = ((0, 0), (0, 0))

    for r in range(rows):
        for c in range(cols):
            heights[c] = heights[c] + 1 if matrix[r][c] == 1 else 0

        stack = []
        for i in range(cols + 1):
            h = heights[i] if i < cols else 0
            while stack and heights[stack[-1]] >= h:
                height = heights[stack.pop()]
                width = i if not stack else i - stack[-1] - 1
                area = height * width

                if area > max_area:
                    max_area = area
                    x1 = stack[-1] + 1 if stack else 0
                    x2 = i - 1
                    y2 = r
                    y1 = r - height + 1
                    best_coords = ((x1, y1), (x2, y2))
            stack.append(i)

    return max_area, best_coords[0], best_coords[1]

def process_glass_laser_cut(image_path, output_path="laser_cut_output.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not open image at {image_path}")

    h, w = img.shape[:2]

    # Convert to Grayscale & Detect Defects
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # Dilate edges slightly to create a safety margin around defects
    kernel = np.ones((5, 5), np.uint8)
    dilated_defects = cv2.dilate(edges, kernel, iterations=1)

    # Create Binary Matrix (1 = Clean Glass, 0 = Defect / Edge Zone)
    scale = max(1, min(h, w) // 100)
    small_h, small_w = h // scale, w // scale
    small_defects = cv2.resize(dilated_defects, (small_w, small_h), interpolation=cv2.INTER_AREA)
    binary_matrix = (small_defects == 0).astype(int)

    # Compute Maximum Intact Rectangle
    area, (x1_s, y1_s), (x2_s, y2_s) = find_max_rectangle(binary_matrix)

    # Scale coordinates back to original image dimensions
    x1, y1 = x1_s * scale, y1_s * scale
    x2, y2 = x2_s * scale, y2_s * scale

    # Render Laser Cutting Trajectory & Visual Overlays
    output_img = img.copy()

    overlay = output_img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    output_img = cv2.addWeighted(overlay, 0.4, output_img, 0.6, 0)

    cv2.rectangle(output_img, (x1, y1), (x2, y2), (255, 255, 255), -1)
    output_img = cv2.addWeighted(img, 0.7, output_img, 0.3, 0)

    # Draw Red Defect Overlay
    defect_mask = cv2.resize(dilated_defects, (w, h))
    output_img[defect_mask > 0] = [0, 0, 255]

    # Draw Laser Trajectory
    cv2.rectangle(output_img, (x1, y1), (x2, y2), (255, 255, 0), 4)
    cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Draw Cutting Direction Trajectory Arrows
    arrow_color = (0, 255, 255)
    cv2.arrowedLine(output_img, (x1, y1), (x2, y1), arrow_color, 2, tipLength=0.05)
    cv2.arrowedLine(output_img, (x2, y1), (x2, y2), arrow_color, 2, tipLength=0.05)
    cv2.arrowedLine(output_img, (x2, y2), (x1, y2), arrow_color, 2, tipLength=0.05)
    cv2.arrowedLine(output_img, (x1, y2), (x1, y1), arrow_color, 2, tipLength=0.05)

    cv2.putText(output_img, "LASER CUT TRAJECTORY", (x1 + 10, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imwrite(output_path, output_img)
    return output_path, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]