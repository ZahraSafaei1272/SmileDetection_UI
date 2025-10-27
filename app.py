from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import cv2
import joblib
import os
from feature_extraction import extract_feature

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model = joblib.load("svm_model.joblib")
scaler = joblib.load("scaler.joblib")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    upload_dir = "static/uploaded"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_classifier.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        result = "No face detected"
    else:
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        img = cv2.resize(image[y:y+h, x:x+w], (256, 256))
        img = [img]
        features = extract_feature(img)
        features = scaler.transform(features)

        prediction = model.predict(features)[0]
        result = "Smiling 😄" if prediction == 1 else "Not Smiling 😐"

    #label = "Smiling" if prediction == 1 else "Not Smiling"
    #color = (0, 255, 0) if prediction == 1 else (0, 0, 255)
    #cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    #cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)



    output_path = "static/output.jpg"
    cv2.imwrite(output_path, image)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "results": result,
        "output_path": "/static/output.jpg"
    })


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
