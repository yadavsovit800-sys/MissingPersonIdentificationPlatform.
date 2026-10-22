import os
from functools import lru_cache
from pathlib import Path

# Keep ONNX/OpenMP from creating many worker threads on small free-tier instances.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import insightface
import numpy as np


@lru_cache(maxsize=1)
def load_face_model():
    # Put the model cache under the same configurable data directory as the app.
    # On Render, set FINDSAFE_DATA_DIR=/var/data and mount a persistent disk there.
    data_dir = Path(os.environ.get("FINDSAFE_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
    model_root = data_dir / "insightface"
    model_root.mkdir(parents=True, exist_ok=True)
    model = insightface.app.FaceAnalysis(
        name="buffalo_s",
        root=str(model_root),
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
    )
    # Smaller detector input reduces peak memory on small hosted instances.
    model.prepare(ctx_id=-1, det_size=(256, 256))
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
