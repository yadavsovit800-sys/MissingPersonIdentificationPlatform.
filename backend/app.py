from pathlib import Path
import uuid
import logging

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .database import init_db, insert_person, get_all_persons, get_person_by_vector_id
from .face_embedding import get_embedding
from .vector_db import ensure_collection, insert_vector, delete_vector, search_vectors
from .social_search import search_social_media

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__import__("os").environ.get("FINDSAFE_DATA_DIR", str(ROOT / "data")))
IMAGE_DIR = DATA_DIR / "images"
FRONTEND_FILE = ROOT / "frontend" / "index.html"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Missing Person Identification API", version="1.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def startup():
    init_db()
    ensure_collection()
    logger.info("Application storage initialized at %s", DATA_DIR)


@app.get("/")
def home():
    return FileResponse(FRONTEND_FILE) if FRONTEND_FILE.exists() else {"message": "Missing Person Identification API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


async def save_upload(image: UploadFile, destination: Path):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded image is empty")
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller")
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image")
    destination.write_bytes(data)


@app.post("/register")
async def register_person(full_name: str = Form(...), age: int = Form(...), contact_number: str = Form(""), last_seen_location: str = Form(""), image: UploadFile = File(...)):
    if not full_name.strip():
        raise HTTPException(status_code=400, detail="Full name is required")
    if age < 0 or age > 120:
        raise HTTPException(status_code=400, detail="Age must be between 0 and 120")
    case_id = str(uuid.uuid4())
    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".jpg"
    image_path = IMAGE_DIR / f"{case_id}{suffix}"
    vector_id = None
    try:
        await save_upload(image, image_path)
        embedding = get_embedding(str(image_path))
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected. Upload a clear photo showing one face.")
        vector_id = insert_vector(embedding)
        try:
            insert_person(case_id, full_name.strip(), age, contact_number.strip(), last_seen_location.strip(), str(image_path), str(vector_id))
        except Exception:
            try:
                delete_vector(str(vector_id))
            except Exception:
                logger.exception("Vector rollback failed for case %s", case_id)
            raise
        logger.info("Registered case %s", case_id)
        return {"message": "Person registered successfully", "case_id": case_id, "full_name": full_name.strip(), "vector_id": str(vector_id), "vector_size": len(embedding)}
    except HTTPException:
        image_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        logger.exception("Registration failed for case %s", case_id)
        image_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Registration failed: {type(exc).__name__}: {str(exc)[:300]}") from exc
    finally:
        await image.close()


@app.post("/search")
async def search_person(image: UploadFile = File(...)):
    temp_path = IMAGE_DIR / f"search_{uuid.uuid4()}.jpg"
    try:
        await save_upload(image, temp_path)
        embedding = get_embedding(str(temp_path))
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected. Upload a clear photo showing one face.")
        matches = search_vectors(embedding, limit=20)
        results = []
        for match in matches:
            person = get_person_by_vector_id(str(match.id))
            if person:
                results.append({"similarity": float(match.score), "person": person})
        results.sort(key=lambda item: item["similarity"], reverse=True)
        return {"message": "Search completed", "matches": results[:5]}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Face search failed")
        raise HTTPException(status_code=500, detail=f"Face search failed: {type(exc).__name__}: {str(exc)[:300]}") from exc
    finally:
        temp_path.unlink(missing_ok=True)
        await image.close()


@app.get("/social-search")
def social_search(name: str, location: str = ""):
    if not name.strip():
        raise HTTPException(status_code=400, detail="Name is required")
    return {"query": name.strip(), "location": location.strip(), "results": search_social_media(name.strip(), location.strip())}


@app.get("/persons")
def persons():
    people = get_all_persons()
    return {"count": len(people), "persons": people}
