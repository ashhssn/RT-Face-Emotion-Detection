import cv2
import numpy as np
from custom_models.dan import DAN
import torch
from torchvision import transforms
from PIL import Image

face_cascade = cv2.CascadeClassifier('src/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_transforms = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                ])

model = DAN(num_head=4, num_class=7, pretrained=False)
model.to(device)
checkpoint = torch.load('checkpoints/affecnet7_epoch6_acc0.6569.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.eval()

while True:
    ret, img = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=7)

    for (x, y, w, h) in faces:
        
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # get roi
        roi = img[y:y + h, x:x + w]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = Image.fromarray(roi)

        # preprocess according to paper
        roi_trf = data_transforms(roi).to(device)
        roi_trf = roi_trf.view(1, 3, 224, 224)

        with torch.no_grad():
            out, _, _ = model(roi_trf)
            _, pred = torch.max(out, 1)
            label = labels[int(pred)]
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