from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tensorflow.keras.models import load_model


class EmotionPredictor:
    EMOTION_LABELS = [
        "Angry",
        "Disgust",
        "Fear",
        "Happy",
        "Normal",
        "Sad",
        "Suprise",
    ]

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.model = None

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                "Copy best_sentiment_model.h5 into the models/ folder."
            )

        print(f"Loading model from: {self.model_path}")
        self.model = load_model(str(self.model_path))
        print("Model loaded successfully.")

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise RuntimeError("Could not load Haar cascade classifier.")

    def predict_one_face(self, gray_face: np.ndarray) -> tuple[str, float]:
        roi = cv2.resize(gray_face, (48, 48))
        roi = roi.astype("float32") / 255.0

        # Your original model code uses shape:
        # (batch, 48, 48)
        roi = np.expand_dims(roi, axis=0)

        prediction = self.model.predict(roi, verbose=False)
        prediction = np.asarray(prediction)

        # Handles a normal 7-class output.
        probabilities = prediction[0]
        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])

        label = self.EMOTION_LABELS[class_index]
        return label, confidence

    def predict(self, frame: np.ndarray) -> dict[str, Any]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        results = []

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]

            if roi.size == 0:
                continue

            label, confidence = self.predict_one_face(roi)

            results.append({
                "emotion": label,
                "confidence": round(confidence, 4),
                "box": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                }
            })

        # The frontend receives the original image dimensions so that
        # it can scale the returned coordinates onto the canvas.
        height, width = frame.shape[:2]

        return {
            "success": True,
            "faces_detected": len(results),
            "image_width": width,
            "image_height": height,
            "faces": results
        }
