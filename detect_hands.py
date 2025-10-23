import numpy as np
import cv2

def detect_hands_raised(keypoints, frame):
    """
    keypoints: np.ndarray of shape (N, 17, 2)
    frame: image array for annotation
    returns: list of booleans for each person
    """
    raised_flags = []

    for kp in keypoints:
        # get keypoints safely
        # https://docs.ultralytics.com/tasks/pose/
        try:
            nose, l_wrist, r_wrist = kp[0], kp[9], kp[10]
        except IndexError:
            raised_flags.append(False)
            continue

        def hand_up(nose, wrist):
            if np.any(np.isnan([*nose, *wrist])):
                return False
            return wrist[1] < nose[1]

        left_up = hand_up(nose, l_wrist)
        right_up = hand_up(nose, r_wrist)
        raised = left_up or right_up
        raised_flags.append(raised)

        if left_up and right_up:
            text = "Beide"
            coords = (int((l_wrist[0]+r_wrist[0])/2), int((l_wrist[1]+r_wrist[1])/2))
        elif left_up:
            text = "Links"
            coords = (int(l_wrist[0]), int(l_wrist[1]))
        elif right_up:
            text = "Rechts"
            coords = (int(r_wrist[0]), int(r_wrist[1]))

        if raised:
            cv2.putText( frame, text, coords, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return raised_flags
