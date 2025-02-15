import cv2
import numpy as np
from tensorflow.keras.models import load_model
from custom_models.separableconv import SeparableConv2DWrapper

# Load the model with custom_objects
emotion_model = load_model(
    'weights/bestXceptionPlusData.h5',
    custom_objects={'SeparableConv2D': SeparableConv2DWrapper}
)

face_cascade = cv2.CascadeClassifier('src/haarcascade_frontalface_default.xml')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=7)
    # Calculate FPS
    if 'prev_time' not in globals():
        prev_time = cv2.getTickCount()
    else:
        current_time = cv2.getTickCount()
        time_diff = (current_time - prev_time) / cv2.getTickFrequency()
        fps = 1.0 / time_diff
        prev_time = current_time
        # Display FPS on the image
        cv2.putText(img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, (255, 255, 255), 2)

    for (x, y, w, h) in faces:
        
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # extract ROI
        roi_gray = gray[y:y + h, x:x + w]
        try:
            # resize ROI to match the model's expected input size (example: 48x48)
            roi_gray = cv2.resize(roi_gray, (48, 48))
        except Exception as e:
            continue

        # Normalize and reshape for the model
        roi = roi_gray.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)      # Batch size axis
        roi = np.expand_dims(roi, axis=-1)     # Channel axis for grayscale

        # Predict emotion
        preds = emotion_model.predict(roi)[0]
        emotion_index = np.argmax(preds)
        label = emotions[emotion_index]

        # Annotate the image with the emotion label
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (36, 255, 12), 2)

    # Display the processed image
    cv2.imshow('img', img)
    
    # Stop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()