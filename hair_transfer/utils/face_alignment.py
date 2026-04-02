"""
Face alignment utility.

Detects faces, finds 68 landmarks, and crops/aligns to a
canonical 256×256 face crop suitable for StyleGAN2.

Requires: face_alignment (pip install face-alignment)
"""
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


class FaceAligner:
    """
    Aligns faces to a canonical crop using 68-point landmarks.

    Usage:
        aligner = FaceAligner(output_size=256)
        aligned_pil = aligner.align(pil_image)
    """

    def __init__(self, output_size: int = 256, device: str = "cpu"):
        import face_alignment
        self.output_size = output_size
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=device,
            flip_input=False,
        )

    def align(self, img: Image.Image) -> Image.Image | None:
        """
        Align a PIL image. Returns aligned PIL image or None if no face found.
        """
        # Resize large images before detection to save memory
        max_size = 1024
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        arr = np.array(img.convert("RGB"))
        preds = self.fa.get_landmarks(arr)
        if preds is None or len(preds) == 0:
            return None

        lm = preds[0]   # [68, 2]  (x, y)
        return self._crop_from_landmarks(img, lm)

    def _crop_from_landmarks(self, img: Image.Image,
                              lm: np.ndarray) -> Image.Image:
        """Compute similarity transform from landmarks and warp image."""
        # Eye and mouth keypoints
        eye_left  = lm[36:42].mean(0)
        eye_right = lm[42:48].mean(0)
        mouth_l   = lm[48]
        mouth_r   = lm[54]

        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_avg    = (mouth_l + mouth_r) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Canonical crop rectangle
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x) + 1e-8
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1

        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # Shrink if needed
        shrink = int(np.floor(qsize / self.output_size * 0.5))
        if shrink > 1:
            rsize = (img.width // shrink, img.height // shrink)
            img   = img.resize(rsize, Image.LANCZOS)
            quad /= shrink
            qsize /= shrink

        # Crop with padding
        border = max(int(np.round(qsize * 0.1)), 3)
        crop   = (
            int(np.floor(min(quad[:, 0]))) - border,
            int(np.floor(min(quad[:, 1]))) - border,
            int(np.ceil(max(quad[:, 0])))  + border,
            int(np.ceil(max(quad[:, 1])))  + border,
        )
        crop = (
            max(crop[0], 0), max(crop[1], 0),
            min(crop[2], img.width), min(crop[3], img.height),
        )
        img  = img.crop(crop)
        quad -= np.array([crop[0], crop[1]])

        # Warp to square
        img = img.transform(
            (self.output_size, self.output_size),
            Image.QUAD,
            (quad + 0.5).flatten(),
            Image.BILINEAR,
        )
        return img

    def align_batch(self, images: list[Image.Image]) -> list[Image.Image | None]:
        return [self.align(img) for img in images]
