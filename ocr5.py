import cv2
import pytesseract

caminho = r"C:\Users\paulo.mendonca\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = caminho

# Localização do diretório onde os arquivos .txt com as bounding boxes estão salvos
labels_dir = "runs/detect/exp6/labels/"

# Nome da imagem que você quer processar
image_name = "corrida4.jpg"

# Ler as coordenadas das bounding boxes a partir do arquivo .txt correspondente
with open(labels_dir + image_name.replace(".jpg", ".txt"), 'r') as f:
    boxes = f.readlines()

# Carregar a imagem original
image = cv2.imread("../fotos/" + image_name)

# Processar cada bounding box
for box in boxes:
    # As coordenadas no arquivo estão no formato [class x_center y_center width height]
    _, x_center, y_center, width, height = map(float, box.strip().split())

    # Converta as coordenadas do centro para coordenadas de canto
    x1, y1 = int((x_center - width / 2) * image.shape[1]), int((y_center - height / 2) * image.shape[0])
    x2, y2 = int((x_center + width / 2) * image.shape[1]), int((y_center + height / 2) * image.shape[0])

    # Desenhar a bounding box na imagem original
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Cortar a bounding box da imagem
    cropped = image[y1:y2, x1:x2]

    # Usar pytesseract para extrair texto
    text = pytesseract.image_to_string(cropped)
    print(text)

# Mostrar a imagem original com as bounding boxes
cv2.imshow('Image with Bounding Boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
