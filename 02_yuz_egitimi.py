import cv2
import numpy as np
from PIL import Image
import os

# Verilerin yolu
path = 'veriseti'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")


# imajların alınması ve etiketlenmesi için fonksiyon
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    ornekler = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # gri
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[0])

        yuzler = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in yuzler:
            ornekler.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)
    return ornekler, ids


print("\n [INFO] yuzler eğitiliyor. Biraz bekleyin ...")
yuzler, ids = getImagesAndLabels(path)
recognizer.train(yuzler, np.array(ids))
# Modeli egitim/egitim.yml dosyasına kaydet
recognizer.write('egitim/egitim.yml')
# Eğitilen yüz sayısını göster ve kodu sonlandır
print(f"\n [INFO] {len(np.unique(ids))} yüz eğitildi. Betik sonlandırılıyor.")