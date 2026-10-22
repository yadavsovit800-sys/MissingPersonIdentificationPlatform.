# Missing Person Identification Platform

## About the Project
The Missing Person Identification Platform is a student project designed to help organize missing-person records and search for possible matches using face-image similarity. It combines a web interface, a Python backend, a database, and a vector search system.

**Note:** A face match is only a possible lead. It does not prove a person's identity and must be verified by authorized people.

## Features
- Register a missing-person record.
- Store record details in SQLite.
- Generate 512-dimensional face embeddings using InsightFace.
- Store and search face embeddings using Qdrant.
- Search for possible matches from an uploaded image.
- Provide a web interface for interacting with the backend.
- Expose API endpoints through FastAPI.

## Technologies Used
- **Python** — backend programming
- **FastAPI** — REST API
- **SQLite** — record storage
- **Qdrant** — vector similarity search
- **InsightFace** — face detection and embeddings
- **HTML** — frontend
- **Uvicorn** — development server

## Project Structure
```text
MissingPersonIdentificationPlatform/
├── backend/
│   ├── app.py
│   ├── database.py
│   ├── face_embedding.py
│   ├── vector_db.py
│   ├── social_search.py
│   └── fix_backend.py
├── frontend/
│   └── index.html
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Install Python and Git
Install a compatible version of Python and Git on your computer.

### 2. Open the project folder
Open the project in VS Code and open a terminal in the project directory.

### 3. Create and activate a virtual environment (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked by your PowerShell policy, you can use the environment's Python directly instead.

### 4. Install dependencies
If a `requirements.txt` file is available, run:
```powershell
python -m pip install -r requirements.txt
```

Otherwise, install the packages required by the backend according to its imports and compatible InsightFace/ONNX Runtime versions.

### 5. Start the backend
From the project root:
```powershell
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open the API documentation at:
http://127.0.0.1:8000/docs

### 6. Start the frontend
Open a second terminal and run:
```powershell
python -m http.server 5500 --directory frontend
```

Open the interface at:
http://127.0.0.1:5500/index.html

Keep both terminals running while testing.

## Testing
1. Open the frontend and check that it loads.
2. Register a test record using fictional information and a test image you are allowed to use.
3. Search with a test image and check whether the API returns possible matches.
4. Check the backend health endpoint at:
   http://127.0.0.1:8000/health
5. Check the API documentation for available endpoints and responses.

## Privacy and Responsible Use
- Do not publish real missing-person photographs or personal contact details in a public repository.
- Do not commit databases, uploaded images, credentials, API keys, or `.env` files.
- Use fictional data or authorized test images during development.
- Restrict access to records and images in any real deployment.
- Face similarity can produce false matches. Verify all leads through appropriate procedures and authorized authorities.

## Project Status
This is a student project for learning and demonstration. Its features and deployment security should be tested before real-world use.
