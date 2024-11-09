from deepface import DeepFace
import numpy as np
import cv2
import pandas as pd

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    filename = "Frame.png"
    cv2.imwrite(filename, frame)

    cv2.imshow("Frame", frame)
    

    if cv2.waitKey(1) == ord("q"):
        break
    
    result = DeepFace.find(
    img_path = frame,
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
        print(str(match_df['identity'][0]))[2:8]
        break
    

cap.release()
cv2.destroyAllWindows()
