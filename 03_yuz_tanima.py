import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('egitim/egitim.yml')
cascadePath = "cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
names = ['None', 'Omer', 'Ahmet', 'Ali', 'Ayşe']

# Canlı video yakalamayı başlat
kamera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
kamera.set(3, 1200)  # video genişliğini belirle
kamera.set(4, 1000)  # video yüksekliğini belirle
# minimum pencere boyutunu belirle
minW = 0.1 * kamera.get(3)  # genişlik
minH = 0.1 * kamera.get(4)  # yükseklik
while True:
    ret, img = kamera.read()
    gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    yuzler = faceCascade.detectMultiScale(
        gri,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    for (x, y, w, h) in yuzler:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        id, uyum = recognizer.predict(gri[y:y + h, x:x + w])
        if (uyum < 100):
            id = names[id]
            uyum = f"Uyum=  {round(uyum, 0)}%"
        else:
            id = "bilinmiyor"
            uyum = f"Uyum=  {round(uyum, 0)}%"
        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (0, 0, 255), 1)
        cv2.putText(img, str(uyum), (x + 5, y + h + 25), font, 1, (0, 0, 255), 1)
    cv2.imshow('kamera', img)
    k = cv2.waitKey(10) # Çıkış için Esc veya q tuşu
    if k == 27:
        break
# Belleği temizle
kamera.release()
cv2.destroyAllWindows()