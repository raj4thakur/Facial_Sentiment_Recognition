from pathlib import Path

from tensorflow.keras.models import load_model

model_path = Path("models/best_sentiment_model.h5")

if not model_path.exists():
    raise FileNotFoundError(
        "Put best_sentiment_model.h5 inside the models folder first."
    )

model = load_model(str(model_path))

print("Model loaded successfully")
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)
print("Model output classes should match 7 emotion labels.")
