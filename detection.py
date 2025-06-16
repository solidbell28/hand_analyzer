import math
import numpy as np
import mediapipe as mp
import cv2
import os

THRESHOLD = 0.98

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils


def cosine_similarity(v1, v2):
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    norm1 = math.hypot(v1[0], v1[1])
    norm2 = math.hypot(v2[0], v2[1])
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)


def parse_img(img_path: str, output_txt: str, original_img_path: str):
    image = cv2.imread(img_path)
    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        raise ValueError('Hand not detected')
    hand_lms = result.multi_hand_landmarks[0]

    lm_coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms.landmark]

    tip_ids = [4, 8, 12, 16, 20]
    base_ids = [2, 5, 9, 13, 17]
    tips = [lm_coords[i] for i in tip_ids]
    bases = [lm_coords[i] for i in base_ids]

    pose_parts = []
    for i in range(len(tips)-1):
        v1 = (tips[i][0] - bases[i][0], tips[i][1] - bases[i][1])
        v2 = (tips[i+1][0] - bases[i+1][0], tips[i+1][1] - bases[i+1][1])
        cos_val = cosine_similarity(v1, v2)
        pose_parts.append('+' if cos_val > THRESHOLD else '-')
    pose_code = ''.join(
        f'{i+1}{pose_parts[i]}' for i in range(len(pose_parts))
    ) + '5'

    valleys = []
    for j in range(len(bases) - 1):
        x_mid = (bases[j][0] + bases[j + 1][0]) / 2
        y_mid = (bases[j][1] + bases[j + 1][1]) / 2
        valleys.append((x_mid, y_mid))

    with open(output_txt, 'w') as f:
        f.write(pose_code + '\n')
        parts = ['!,' + os.path.basename(original_img_path)]
        for x, y in tips:
            parts.append(f'T {x} {y}')
        for x, y in valleys:
            parts.append(f'V {x} {y}')
        parts.append('?')
        f.write(','.join(parts))

    cv2.putText(
        image, pose_code, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    line_pts = []
    for i in range(len(tips)):
        line_pts.append(tips[i])
        if i < len(valleys):
            line_pts.append(valleys[i])
    cv2.polylines(image, [np.array(line_pts, np.int32)], False, (255, 0, 0), 2)

    return image
