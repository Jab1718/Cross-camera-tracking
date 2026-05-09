import cv2
import numpy as np

DARK_THRESHOLD = 80

def is_dark(frame: np.ndarray) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.mean() < DARK_THRESHOLD
