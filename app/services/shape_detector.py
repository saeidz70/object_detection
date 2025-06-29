import cv2
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple
import tempfile

def detect_shapes_from_bytes(image_bytes: bytes) -> Tuple[Dict[str, float], np.ndarray]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(image_bytes)
        tmp.flush()
        image_path = tmp.name

    image = cv2.imread(image_path)
    output_image = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    shape_counts = defaultdict(int)
    total_valid = 0

    for c in contours:
        if cv2.contourArea(c) > 100:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)

            if len(approx) == 3:
                shape = "triangle"
            elif len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                shape = "square" if 0.95 <= w / float(h) <= 1.05 else "rectangle"
            elif len(approx) == 5:
                shape = "pentagon"
            else:
                shape = "circle"

            shape_counts[shape] += 1
            total_valid += 1

            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.drawContours(output_image, [c], -1, (0, 255, 0), 2)
                cv2.putText(output_image, shape, (cx - 20, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    shape_stats = {
        shape: round((count / total_valid) * 100, 1)
        for shape, count in shape_counts.items()
    }

    return shape_stats, output_image
