import cv2
import torch
from PIL import Image
import easyocr
import numpy as np
import pandas as pd
import re
import os
import openpyxl

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./runs/train/exp/weights/best.pt')

# Configuração do modelo
model.conf = 0.25

# Create a reader object for easyocr
reader = easyocr.Reader(['en'])  # you can specify other languages as well

# Create a DataFrame to store the results
final_df = pd.DataFrame(columns=['File Name', 'Numbers'])

# Iterar sobre todas as imagens na pasta
directory = '../fotos/'
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        filepath = os.path.join(directory, filename)
        im1 = Image.open(filepath)  # PIL image

        # Inference
        results = model(im1, size=640)  # batch of images

        # Convert PIL image to OpenCV format
        im0 = cv2.cvtColor(np.array(im1), cv2.COLOR_RGB2BGR)

        all_numbers = []
        # Extract text for each bounding box
        for *xyxy, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, xyxy)
            roi = im0[y1:y2, x1:x2]
            text_results = reader.readtext(roi)
            texts = [detection[1] for detection in text_results]
            text = ' '.join(texts)  # Join the detected text pieces (if there are multiple)

            # Extract numbers
            numbers_found = re.findall(r'\d+', text)
            filtered_numbers = []
            for num in numbers_found:
                if f"{num}/" not in text and f"/{num}" not in text:
                    filtered_numbers.append(num)

            all_numbers.extend(filtered_numbers)

        # Append results to the dataframe
        final_df = final_df._append({'File Name': filename, 'Numbers': ' '.join(all_numbers)}, ignore_index=True)

# Save to Excel
final_df.to_excel("results.xlsx", index=False)
print("Results saved to results.xlsx")
