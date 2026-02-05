
import cv2
import numpy as np

cap = cv2.VideoCapture(1)

COLOR_RANGES = {
    "RED": [
        ((0, 60, 40), (10, 255, 255)),
        ((170, 60, 40), (180, 255, 255)),
    ],
    "GREEN": [((30, 25, 30), (90, 255, 255))],
    "BLUE": [((90, 40, 40), (130, 255, 255))],
    "WHITE": [((0, 0, 160), (180, 130, 255))],
    "BLACK": [((0, 0, 0), (180, 120, 120))],
}

COLOR_NAMES_RU = {
    "RED": "красный",
    "GREEN": "зелёный",
    "BLUE": "синий",
    "WHITE": "белый",
}

DRAW_COLORS = {
    "RED": (0, 0, 255),
    "GREEN": (0, 255, 0),
    "BLUE": (255, 0, 0),
    "WHITE": (255, 255, 255),
    "UNKNOWN": (0, 0, 255),
}

MIN_AREA = 1200
ASPECT_MIN = 2.2
ASPECT_MAX = 4.5
COLOR_RATIO_THRESHOLD = 0.22
WHITE_RATIO_THRESHOLD = 0.15
BLACK_BAND_RATIO = 0.04
CANDIDATE_OVERLAP = 0.3
MIN_EXTENT = 0.5
MIN_SATURATION = 60
WHITE_REJECT_RATIO = 0.35
BLACK_BAND_MIN_RATIO = 0.015
BLACK_BAND_VERTICAL_GAP = 0.25
BLACK_BAND_MIN_AREA = 60
BLACK_BAND_MIN_ASPECT = 2.0
BLACK_BAND_X_OVERLAP = 0.4
BLACK_BAND_HEIGHT_RATIO = 0.15
BLACK_BAND_WIDTH_SIMILARITY = 0.6

COLOR_TTL = 15
WHITE_TTL = 30
HIT_CONFIRM = 2
HIT_DECAY = 1
last_seen = {}
color_hits = {"RED": 0, "GREEN": 0, "BLUE": 0, "WHITE": 0}

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def build_mask(hsv_roi, color):
    mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
    for low, high in COLOR_RANGES[color]:
        mask |= cv2.inRange(hsv_roi, np.array(low), np.array(high))
    return mask


def has_black_bands(hsv_roi):
    mask_black = build_mask(hsv_roi, "BLACK")
    height, width = mask_black.shape[:2]
    band = max(1, int(height * 0.18))
    top_band = mask_black[:band, :]
    bottom_band = mask_black[-band:, :]
    top_ratio = cv2.countNonZero(top_band) / top_band.size
    bottom_ratio = cv2.countNonZero(bottom_band) / bottom_band.size

    if top_ratio < BLACK_BAND_MIN_RATIO or bottom_ratio < BLACK_BAND_MIN_RATIO:
        return False

    ys, xs = np.where(mask_black > 0)
    if ys.size == 0:
        return False
    x_left = np.min(xs)
    x_right = np.max(xs)
    if (x_right - x_left) / max(1, width) < 0.4:
        return False

    top_center = np.mean(np.where(top_band > 0)[0]) if top_ratio > 0 else 0
    bottom_center = (
        np.mean(np.where(bottom_band > 0)[0]) + (height - band)
        if bottom_ratio > 0
        else height
    )
    gap_ratio = (bottom_center - top_center) / max(1, height)
    return gap_ratio >= BLACK_BAND_VERTICAL_GAP


def pair_black_bands(bands, frame_height):
    pairs = []
    for i, top in enumerate(bands):
        for j, bottom in enumerate(bands):
            if j == i:
                continue
            if bottom[1] <= top[1]:
                continue
            width_ratio = min(top[2], bottom[2]) / max(1, max(top[2], bottom[2]))
            if width_ratio < BLACK_BAND_WIDTH_SIMILARITY:
                continue
            x_left = max(top[0], bottom[0])
            x_right = min(top[0] + top[2], bottom[0] + bottom[2])
            overlap = max(0, x_right - x_left)
            min_width = min(top[2], bottom[2])
            if min_width == 0:
                continue
            if overlap / min_width < BLACK_BAND_X_OVERLAP:
                continue
            vertical_gap = bottom[1] - (top[1] + top[3])
            if vertical_gap < 0:
                continue
            if (bottom[1] + bottom[3] - top[1]) / max(1, frame_height) < (
                BLACK_BAND_VERTICAL_GAP
            ):
                continue
            pairs.append((top, bottom))
    return pairs


def detect_color(hsv_roi, color_hint):
    total_pixels = hsv_roi.shape[0] * hsv_roi.shape[1]
    if total_pixels == 0:
        return "UNKNOWN"

    white_mask = build_mask(hsv_roi, "WHITE")
    white_ratio = cv2.countNonZero(white_mask) / total_pixels

    if color_hint in ("RED", "GREEN", "BLUE"):
        mask = build_mask(hsv_roi, color_hint)
        ratio = cv2.countNonZero(mask) / total_pixels
        saturation_mean = float(np.mean(hsv_roi[:, :, 1]))
        min_saturation = MIN_SATURATION + (10 if color_hint == "BLUE" else 0)
        if (
            ratio >= COLOR_RATIO_THRESHOLD
            and saturation_mean >= min_saturation
            and white_ratio < WHITE_REJECT_RATIO
        ):
            return color_hint
        return "UNKNOWN"

    if color_hint == "WHITE":
        if white_ratio >= WHITE_RATIO_THRESHOLD and has_black_bands(hsv_roi):
            return "WHITE"
        return "UNKNOWN"

    return "UNKNOWN"


def boxes_overlap(box_a, box_b):
    ax, ay, aw, ah = box_a[:4]
    bx, by, bw, bh = box_b[:4]
    x_left = max(ax, bx)
    y_top = max(ay, by)
    x_right = min(ax + aw, bx + bw)
    y_bottom = min(ay + ah, by + bh)
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def dedupe_boxes(boxes, overlap_threshold=CANDIDATE_OVERLAP):
    result = []
    for box in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
        if any(boxes_overlap(box, kept) > overlap_threshold for kept in result):
            continue
        result.append(box)
    return result


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = clahe.apply(v)
    hsv = cv2.merge([h, s, v])

    frame_area = frame.shape[0] * frame.shape[1]
    max_area = frame_area * 0.2
    candidates = []

    for color in ("RED", "GREEN", "BLUE"):
        mask = build_mask(hsv, color)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0:
                continue
            aspect = h / w
            extent = area / float(w * h)
            if (
                ASPECT_MIN < aspect < ASPECT_MAX
                and h > w
                and extent >= MIN_EXTENT
            ):
                candidates.append((x, y, w, h, color))

    black_mask = build_mask(hsv, "BLACK")
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    black_mask = cv2.dilate(black_mask, np.ones((3, 3), np.uint8), iterations=1)
    black_contours, _ = cv2.findContours(
        black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    black_bands = []
    for cnt in black_contours:
        area = cv2.contourArea(cnt)
        if area < BLACK_BAND_MIN_AREA or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0:
            continue
        if (w / max(1, h)) < BLACK_BAND_MIN_ASPECT:
            continue
        if (h / max(1, frame.shape[0])) > BLACK_BAND_HEIGHT_RATIO:
            continue
        black_bands.append((x, y, w, h))

    white_mask = build_mask(hsv, "WHITE")
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    for top, bottom in pair_black_bands(black_bands, frame.shape[0]):
        x = min(top[0], bottom[0])
        y = top[1]
        w = max(top[0] + top[2], bottom[0] + bottom[2]) - x
        h = (bottom[1] + bottom[3]) - y
        if w == 0 or h == 0:
            continue
        aspect = h / w
        if not (ASPECT_MIN < aspect < ASPECT_MAX and h > w):
            continue
        roi_white = white_mask[y : y + h, x : x + w]
        white_ratio = cv2.countNonZero(roi_white) / float(w * h)
        if white_ratio >= WHITE_RATIO_THRESHOLD:
            candidates.append((x, y, w, h, "WHITE"))
    candidates = dedupe_boxes(candidates)

    detected_colors = []
    for x, y, w, h, color_hint in candidates:
        roi = hsv[y : y + h, x : x + w]
        color = detect_color(roi, color_hint)
        if color == "UNKNOWN":
            continue
        detected_colors.append((color, x, y, w, h))

    for color_name in color_hits:
        color_hits[color_name] = max(0, color_hits[color_name] - HIT_DECAY)

    for color, x, y, w, h in detected_colors:
        color_hits[color] += 1
        if color_hits[color] < HIT_CONFIRM:
            continue
        draw_color = DRAW_COLORS.get(color, DRAW_COLORS["UNKNOWN"])
        cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)
        cv2.putText(
            frame,
            color,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            draw_color,
            2,
        )

    for color, _, _, _, _ in detected_colors:
        if color not in last_seen:
            print(f"обнаружен цилиндр цвета {COLOR_NAMES_RU[color]}")
        last_seen[color] = WHITE_TTL if color == "WHITE" else COLOR_TTL

    for color in list(last_seen.keys()):
        last_seen[color] -= 1
        if last_seen[color] <= 0:
            last_seen.pop(color, None)

    cv2.imshow("Cylinder detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
