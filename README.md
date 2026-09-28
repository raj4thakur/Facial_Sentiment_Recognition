# Facial Expression Recognition - FastAPI + HTML

This project deploys a TensorFlow/Keras facial-expression model behind a FastAPI API and provides a browser-based webcam frontend.

## 1. Project structure

```text
emotion_fastapi_project/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── predictor.py
│   │
│   ├── templates/
│   │   └── index.html
│   │
│   └── static/
│       ├── style.css
│       └── script.js
│
├── models/
│   ├── best_sentiment_model.h5
│   └── PUT_MODEL_HERE.txt
│
├── requirements.txt
├── render.yaml
├── .gitignore
└── README.md
```

## 2. Put your model in the correct location

Copy:

```text
best_sentiment_model.h5
```

to:

```text
models/best_sentiment_model.h5
```

The application expects the model to have the same input format as the code you provided:

```python
roi = cv2.resize(roi, (48, 48))
roi = roi.astype("float32") / 255.0
roi = np.expand_dims(roi, axis=0)
```

The expected output is seven classes in this order:

```text
0 = Angry
1 = Disgust
2 = Fear
3 = Happy
4 = Normal
5 = Sad
6 = Suprise
```

## 3. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.venv\Scripts\Activate.ps1
```

## 4. Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Start the application

```powershell
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

Allow browser camera permission.

## 6. API endpoints

### Home

```text
GET /
```

### Health

```text
GET /health
```

Example:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": ".../models/best_sentiment_model.h5"
}
```

### Prediction

```text
POST /predict
```

The request contains an image under the multipart field:

```text
file
```

Example response:

```json
{
  "success": true,
  "faces_detected": 1,
  "image_width": 640,
  "image_height": 480,
  "faces": [
    {
      "emotion": "Happy",
      "confidence": 0.94,
      "box": {
        "x": 200,
        "y": 100,
        "width": 180,
        "height": 180
      }
    }
  ]
}
```

## 7. Important architecture change

Do NOT use this in FastAPI:

```python
cap = cv2.VideoCapture(0)
```

That opens the webcam attached to the server.

Instead:

```text
User browser
    |
    | getUserMedia()
    v
Browser webcam
    |
    | JPEG frames
    v
FastAPI /predict
    |
    v
OpenCV face detection
    |
    v
Keras model
    |
    v
JSON result
    |
    v
Browser canvas
```

This is what the included frontend implements.

## 8. Deploy on Render

Push the entire project to GitHub.

Make sure the repository contains:

```text
app/
models/best_sentiment_model.h5
requirements.txt
render.yaml
```

Then create a Render Web Service from the GitHub repository.

Build command:

```text
pip install -r requirements.txt
```

Start command:

```text
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Python version:

```text
3.10.13
```

After deployment, Render gives you an HTTPS URL.

Open that URL in a browser and allow camera permission.

## 9. IMPORTANT: model file and GitHub

GitHub has file-size limits. If your `.h5` model is large, do not blindly commit a very large model to GitHub.

Possible production approaches are:

1. Git LFS
2. Hugging Face model storage
3. Cloud object storage
4. Render disk/storage strategy

For a small model, keeping it in the repository can be sufficient.

## 10. Browser camera security

Web browsers generally require a secure context for camera access.

Local development:

```text
http://127.0.0.1:8000
```

is normally allowed.

Production should use:

```text
https://your-app.onrender.com
```

Do not deploy the frontend over plain HTTP and expect browser camera access to work reliably.

## 11. Performance

The browser sends approximately one frame every 400 ms.

That is approximately:

```text
2.5 requests/second
```

This is intentionally much lighter than sending 30 FPS to the server.

If the model is slow, increase:

```javascript
setInterval(sendFrameForPrediction, 700);
```

or:

```javascript
setInterval(sendFrameForPrediction, 1000);
```

## 12. Multiple faces

The API detects every Haar-cascade face and returns:

```text
faces: [...]
```

The frontend draws a box and emotion for every detected face.

## 13. If your model gives an input-shape error

Run:

```python
from tensorflow.keras.models import load_model

model = load_model("models/best_sentiment_model.h5")

print(model.input_shape)
print(model.output_shape)
```

The current application assumes the input is compatible with:

```text
(batch, 48, 48)
```

If your actual model expects:

```text
(batch, 48, 48, 1)
```

change this in `app/predictor.py`:

```python
roi = np.expand_dims(roi, axis=0)
```

to:

```python
roi = np.expand_dims(roi, axis=-1)
roi = np.expand_dims(roi, axis=0)
```

Then the input becomes:

```text
(batch, 48, 48, 1)
```

This depends on the actual model architecture.

## 14. Testing model loading

Before starting FastAPI, you can test:

```powershell
python -c "from tensorflow.keras.models import load_model; m=load_model('models/best_sentiment_model.h5'); print(m.input_shape); print(m.output_shape)"
```

If this prints the expected shapes, start FastAPI.

## 15. API documentation

FastAPI automatically provides:

```text
/docs
```

So locally:

```text
http://127.0.0.1:8000/docs
```

After deployment:

```text
https://your-app.onrender.com/docs
```

You can use Swagger UI there to manually test `/predict`.