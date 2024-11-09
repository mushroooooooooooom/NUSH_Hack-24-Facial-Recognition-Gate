from deepface import DeepFace
import numpy as np
import cv2
import pandas as pd
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    image_data = data['image']

    # Decode base64 image data
    image_data = image_data.split(",")[1]  # Remove 'data:image/jpeg;base64,'
    image_bytes = BytesIO(base64.b64decode(image_data))
    image = Image.open(image_bytes)
    # while True:
#     ret, frame = cap.read()

#     filename = "Frame.png"
#     cv2.imwrite(filename, frame)

#     cv2.imshow("Frame", frame)
    
def face_analyse():
    result = DeepFace.find(
    img_path = "image/jpeg",
    db_path = "db",
    model_name="VGG-Face", 
    distance_metric="cosine", 
    enforce_detection=False
    )
    if len(result) >=1:
        match_df = result[0]
        if not match_df.empty:
            pass
        print(match_df['identity'])
        x = str(match_df['identity'][0])
        result = x
        print(x[3:8])

    # Process the image (e.g., analyze with a model or OpenCV)
    # For example, here we just return a placeholder analysis result

    return jsonify({'analysis': result})

if __name__ == '__main__':
    app.run(debug=True)
#cap = cv2.VideoCapture(0)



        
    


