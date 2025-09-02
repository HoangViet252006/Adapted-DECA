import numpy as np
import torch
class FAN(object):
    def __init__(self, device='cuda'):
        import face_alignment
        if device == 'cpu':
            self.model = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                flip_input=False,
                device=device   # force CPU
            )
        else:
            self.model = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                flip_input=False,
                # dtype=torch.bfloat16,
                device=device  # force GPU
            )


    def run(self, image: np.ndarray):
        """
        Run landmark detection.

        Args:
            image (np.ndarray): RGB image, uint8, shape [H, W, 3].

        Returns:
            tuple: (bbox, landmarks) where
                bbox: [x_min, y_min, x_max, y_max] or [0] if no face.
                landmarks: np.ndarray of shape (68, 2) or None if no face.
        """
        out = self.model.get_landmarks(image)
        if not out:  # handles None and empty list
            return [0], None

        kpt = out[0] # (68, 2)
        x_min, y_min = np.min(kpt, axis=0)
        x_max, y_max = np.max(kpt, axis=0)
        bbox = [x_min, y_min, x_max, y_max]

        return bbox, kpt