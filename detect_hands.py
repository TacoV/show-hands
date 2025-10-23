import numpy as np
import cv2

def detect_hands_raised(keypoints, frame):
    """
    keypoints: np.ndarray of shape (N, 17, 2)
    frame: image array for annotation
    returns: list of booleans for each person
    """
    votes = []

    for kp in keypoints:
        # get keypoints safely
        # https://docs.ultralytics.com/tasks/pose/
        try:
            nose = kp[0]
            l_eye, r_eye = kp[1], kp[2]
            l_wrist, r_wrist = kp[9], kp[10]
        except IndexError:
            continue

        if np.any(np.isnan([*nose, *l_eye, *r_eye, *l_wrist, *r_wrist])):
            continue

        if nose[1] < l_wrist[1] and nose[1] < r_wrist[1]:
            continue

        pro = l_wrist[1] > r_wrist[1]
        con = not pro

        # Draw a green triangle for pro, red for con
        size = 0.6 * (l_eye[0] - r_eye[0])
        leftside = np.array([nose[0] - size, nose[1]], dtype=np.int32)
        rightside = np.array([nose[0] + size, nose[1]], dtype=np.int32)
        point = np.array([nose[0], nose[1] + size * (-2 if pro else 2)], dtype=np.int32)
        cv2.fillPoly(frame,
            [np.array([leftside, rightside, point])],
            color=(0, 255, 0) if pro else (0, 0, 255))

    return votes
