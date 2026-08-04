"""
TFLite Export Script for MobileNetV2 Frame Quality Classifier
Converts the saved Keras model (models/quality_mobilenetv2.keras)
to a flatbuffer .tflite format for mobile client-side Flutter execution.

Outputs:
- models/quality_mobilenetv2.tflite
"""

import os
import tensorflow as tf

def export_to_tflite(
    model_path: str = "models/quality_mobilenetv2.keras",
    tflite_path: str = "models/quality_mobilenetv2.tflite"
) -> bool:
    if not os.path.exists(model_path):
        print(f"[export_tflite] Error: Model file '{model_path}' not found. Run train_quality_classifier.py first.")
        return False

    try:
        print(f"[export_tflite] Loading Keras model from '{model_path}'...")
        model = tf.keras.models.load_model(model_path)

        print("[export_tflite] Converting model to TFLite format...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # Enable standard optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

        os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        size_kb = os.path.getsize(tflite_path) / 1024.0
        print(f"[export_tflite] Successfully exported TFLite model to '{tflite_path}' ({size_kb:.2f} KB).")
        return True
    except Exception as e:
        print(f"[export_tflite] Failed to export TFLite model: {e}")
        return False

if __name__ == "__main__":
    export_to_tflite()
