import cv2
import insightface

app = insightface.app.FaceAnalysis()
app.prepare(ctx_id=0)

def get_embedding(image_path):
    img = cv2.imread(image_path)
    faces = app.get(img)
    if not faces:
        return None
    return faces[0].embedding.tolist()

embedding = get_embedding("images/person.jpg")

if embedding:
    print("Face embedding created")
    print("Vector size:", len(embedding))
else:
    print("No face detected")
