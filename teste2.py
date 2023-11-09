import cv2
import torch
from PIL import Image
import easyocr
import numpy as np
import pandas as pd
import re

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./runs/train/exp/weights/custom_model.pt')

# Images
im1 = Image.open('../fotos/corrida11.jpg')  # PIL image

# Inference
model.conf = 0.25
results = model(im1, size=640)  # batch of images

# Convert PIL image to OpenCV format
im0 = cv2.cvtColor(np.array(im1), cv2.COLOR_RGB2BGR)

# Create a reader object for easyocr
reader = easyocr.Reader(['en'])  # you can specify other languages as well

# Create a DataFrame to store the results
df = pd.DataFrame(columns=['Bounding Box', 'Text'])

numbers= []

# Extract text for each bounding box
for *xyxy, conf, cls in results.xyxy[0]:
    x1, y1, x2, y2 = map(int, xyxy)
    roi = im0[y1:y2, x1:x2]
    text_results = reader.readtext(roi)
    texts = [detection[1] for detection in text_results]
    text = ' '.join(texts)  # Join the detected text pieces (if there are multiple)

    # Use regular expression to find numbers not part of a date
    numbers2 = ' '.join([match.group() for match in re.finditer(r'(?<!\d/)\d+(?!/\d)', text)])
    numbers.append(numbers2)
    
    # Append results to the dataframe
    df = df._append({'Bounding Box': f'({x1}, {y1}, {x2}, {y2})', 'Text': text}, ignore_index=True)

# Print the table
print('numeros:')
print(numbers)
print(df)

# Show the image with bounding boxes
results.show()

results.xyxy[0]  # im1 predictions (tensor)
results.pandas().xyxy[0]  # im1 predictions (pandas)