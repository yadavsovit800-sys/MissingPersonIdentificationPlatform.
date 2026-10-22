from pathlib import Path
import uuid
import shutil
import logging

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .database import init_db, insert_person, get_all_persons, get_person_by_vector_id
from .face_embedding import get_embedding
from .vector_db import ensure_collection, insert_vector, search_vectors
from .social_search import search_social_media

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "images"
FRONTEND_FILE = ROOT / "frontend" / "index.html"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Missing Person Identification API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    init_db()
    ensure_collection()
    logger.info("Application storage initialized")

@app.get("/")
def home():
    if FRONTEND_FILE.exists():
        return FileResponse(FRONTEND_FILE)
    return {"message": "Missing Person Identification API is running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/register")
async def register_person(
    full_name: str = Form(...),
    age: int = Form(...),
    contact_number: str = Form(""),
    last_seen_location: str = Form(""),
    image: UploadFile = File(...),
):
    if not full_name.strip():
        raise HTTPException(status_code=400, detail="Full name is required")
    if age < 0 or age > 120:
        raise HTTPException(status_code=400, detail="Invalid age")
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image")

    case_id = str(uuid.uuid4())
    image_path = IMAGE_DIR / f"{case_id}{Path(image.filename or '').suffix.lower()}"
    vector_id = None
    try:
        with image_path.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        embedding = get_embedding(str(image_path))
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected in the uploaded image")

        vector_id = insert_vector(embedding)
        insert_person(
            case_id=case_id,
            full_name=full_name.strip(),
            age=age,
            contact_number=contact_number.strip(),
            last_seen_location=last_seen_location.strip(),
            image_path=str(image_path),
            vector_id=str(vector_id),
        )
        return {
            "message": "Person registered successfully",
            "case_id": case_id,
            "full_name": full_name.strip(),
            "vector_id": str(vector_id),
            "vector_size": len(embedding),
        }
    except HTTPException:
        image_path.unlink(missing_ok=True)
        raise
    except Exception:
        logger.exception("Registration failed")
        image_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Registration failed. Check Render logs for the server-side error.")
    finally:
        await image.close()

@app.post("/search")
async def search_person(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image")
    temp_path = IMAGE_DIR / f"search_{uuid.uuid4()}{Path(image.filename or '').suffix.lower()}"
    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        embedding = get_embedding(str(temp_path))
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected in the uploaded image")
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
    except Exception:
        logger.exception("Face search failed")
        raise HTTPException(status_code=500, detail="Face search failed. Check Render logs for the server-side error.")
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
