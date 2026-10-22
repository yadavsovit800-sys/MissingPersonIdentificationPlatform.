from functools import lru_cache
from pathlib import Path

import cv2
import insightface
import numpy as np


@lru_cache(maxsize=1)
def load_face_model():
    model = insightface.app.FaceAnalysis(
        name="buffalo_s",
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"]
    )
    model.prepare(ctx_id=-1, det_size=(320, 320))
    return model


def get_embedding(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    faces = load_face_model().get(image)

    if not faces:
        return None

    return np.asarray(faces[0].embedding, dtype=np.float32).tolist()


if __name__ == "__main__":
    image_path = input("Enter image path: ").strip().strip('"')
    embedding = get_embedding(Path(image_path))

    if embedding is not None:
        print("Face embedding created")
        print("Vector size:", len(embedding))
    else:
        print("No face detected")
