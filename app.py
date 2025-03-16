import os
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

# ✅ Initialize Flask App
app = Flask(__name__)

# ✅ Get the absolute path of the project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Define the correct paths to the model and encoder
model_path = os.path.join(BASE_DIR, "models", "fish_model.pkl")
encoder_path = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

# ✅ Load the trained model and label encoder
try:
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        print("✅ Model and Label Encoder loaded successfully!")
    else:
        print("❌ Model files not found. Please check the 'models/' directory.")
        model = None
        encoder = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    encoder = None

# ✅ Home Route - Load the Frontend Page
@app.route("/")
def home():
    return render_template("index.html")

# ✅ Prediction Route (Handles API and Form Requests)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None or encoder is None:
            return jsonify({"error": "Model files not found. Please train the model first."}), 500

        # Collect input data
        if request.is_json:  # JSON request (for API)
            data = request.get_json()
            features = [float(data[key]) for key in ["Weight", "Length1", "Length2", "Length3", "Height", "Width"]]
        else:  # Form request (from HTML form)
            features = [float(request.form[key]) for key in ["Weight", "Length1", "Length2", "Length3", "Height", "Width"]]

        # Convert to numpy array
        features = np.array(features).reshape(1, -1)

        # Predict the label
        predicted_label = model.predict(features)[0]

        # Convert label back to fish species name
        predicted_species = encoder.inverse_transform([predicted_label])[0]

        # If request is from API, return JSON response
        if request.is_json:
            return jsonify({"prediction": predicted_species})

        # If request is from HTML form, render the result page
        return render_template("result.html", prediction=predicted_species)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ✅ Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
