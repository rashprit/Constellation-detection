from flask import Flask, request, jsonify

import os
import sys

# Ensure the current directory is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from constellation_detector import detect_constellation_from_image

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')  # Add this route for the home page
def home():
    return "Constellation Detector API is running!", 200

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        result = detect_constellation_from_image(file_path)
    except Exception as e:
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)


