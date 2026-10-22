
from pathlib import Path
import uuid
import shutil

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .database import (
    init_db,
    insert_person,
    get_all_persons,
    get_person_by_vector_id
)
from .face_embedding import get_embedding
from .vector_db import ensure_collection, insert_vector, search_vectors
from .social_search import search_social_media

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

init_db()
ensure_collection()

app = FastAPI(
    title="Missing Person Identification API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def home():
    return {"message": "Missing Person Identification API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/register")
async def register_person(
    full_name: str = Form(...),
    age: int = Form(...),
    contact_number: str = Form(""),
    last_seen_location: str = Form(""),
    image: UploadFile = File(...)
):
    if not full_name.strip():
        raise HTTPException(status_code=400, detail="Full name is required")

    if age < 0 or age > 120:
        raise HTTPException(status_code=400, detail="Invalid age")

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image")

    case_id = str(uuid.uuid4())
    image_path = IMAGE_DIR / f"{case_id}.jpg"

    try:
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        embedding = get_embedding(str(image_path))

        if embedding is None:
            image_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail="No face detected in the uploaded image"
            )

        vector_id = insert_vector(embedding)

        insert_person(
            case_id=case_id,
            full_name=full_name.strip(),
            age=age,
            contact_number=contact_number.strip(),
            last_seen_location=last_seen_location.strip(),
            image_path=str(image_path),
            vector_id=vector_id
        )

        return {
            "message": "Person registered successfully",
            "case_id": case_id,
            "full_name": full_name.strip(),
            "vector_id": vector_id,
            "vector_size": len(embedding)
        }

    except HTTPException:
        raise
    except Exception as e:
        image_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Registration failed: {str(e)}"
        )


@app.post("/search")
async def search_person(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image")

    temp_path = IMAGE_DIR / f"search_{uuid.uuid4()}.jpg"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        embedding = get_embedding(str(temp_path))

        if embedding is None:
            raise HTTPException(
                status_code=400,
                detail="No face detected in the uploaded image"
            )

        matches = search_vectors(embedding, limit=20)
        results = []
        seen_people = set()

        for match in matches:
            vector_id = str(match.id)
            person = get_person_by_vector_id(vector_id)

            if not person:
                continue

            name = str(person.get("full_name", "")).strip().casefold()
            age = str(person.get("age", "")).strip()
            contact = str(person.get("contact_number", "")).strip()

            if name and age and contact:
                identity = (name, age, contact)
            else:
                identity = ("vector_id", vector_id)

            if identity in seen_people:
                continue

            seen_people.add(identity)

            results.append({
                "similarity": float(match.score),
                "person": person
            })

        results.sort(
            key=lambda item: item["similarity"],
            reverse=True
        )

        return {
            "message": "Search completed",
            "matches": results[:5]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )
    finally:
        temp_path.unlink(missing_ok=True)


@app.get("/social-search")
def social_search(name: str, location: str = ""):
    if not name.strip():
        raise HTTPException(status_code=400, detail="Name is required")

    return {
        "query": name.strip(),
        "location": location.strip(),
        "results": search_social_media(name.strip(), location.strip())
    }


@app.get("/persons")
def persons():
    all_persons = get_all_persons()
    return {"count": len(all_persons), "persons": all_persons}
