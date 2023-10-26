import torch
from pathlib import Path
import pytesseract
import re
import cv2
import numpy as np

caminho = r"C:\Users\paulo.mendonca\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = caminho

# Carregar o modelo YOLOv5 treinado
from models.experimental import attempt_load
weights_path = './runs/train/exp/weights/best.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(weights_path)
model.eval()

# Diretório com suas imagens
image_dir = Path("../fotos/")

numbers_detected = []

# Processar cada imagem no diretório
for image_path in image_dir.glob("*.jpg"):
    # Carregar a imagem com OpenCV
    # Carregar a imagem com OpenCV
    img = cv2.imread(str(image_path))
    img = cv2.resize(img, (640, 640))  # Redimensionar a imagem
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converta para RGB
    img_tensor = torch.from_numpy(img_rgb).float().div(255.0).permute(2, 0, 1)  # Converta para tensor
    img_tensor = img_tensor.unsqueeze(0)  # Adicione dimensão de lote

    # Fazer detecção com YOLOv5
    results = model(img_tensor)

    # Extrair bounding boxes (x_center, y_center, width, height)
    boxes = results[0][:, :-1].cpu().numpy()
    print(boxes.shape)


    # Convertendo formato (x_center, y_center, width, height) para (x_min, y_min, x_max, y_max)
    boxes[:, :, :2] = boxes[:, :, :2] - boxes[:, :, 2:4] / 2
    boxes[:, :, 2:4] = boxes[:, :, :2] + boxes[:, :, 2:4]

    # Processar cada bounding box
    for box in boxes[0]:
        x1, y1, x2, y2 = map(int, box[:4])


        roi = img[y1:y2, x1:x2]
        
        # Desenhar bounding box na imagem
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Use pytesseract para fazer OCR na ROI
        texto = pytesseract.image_to_string(roi, config='--psm 11')
        numbers = re.findall(r'\b\d+\b', texto)
        numbers_detected.extend(numbers)
    
    # Exibir a imagem
    cv2.imshow('Image with Bounding Boxes', img)
    cv2.waitKey(5000)

print('NUMEROSSS')
print(numbers_detected)

