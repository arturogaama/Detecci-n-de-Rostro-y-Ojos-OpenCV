import numpy as np
import cv2

# Creamos el objeto que permite capturar video desde la camara
cap = cv2.VideoCapture(0)

# Objeto de opencv que identifica los rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Objeto de opencv que identifica ojos
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


while True:
    ret, frame = cap.read()

    # Se convierte la imagen capturada a escala de grises
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Se realiza la detección de rostros en las imagenes en gris
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)


    # Cada que se identifique una cara, vamos a buscar en donde están los ojos, unicamente dentro del la caja de la cara
    for (x, y, w, h) in faces:

        # 'Recortamos' unicamente el area que nos interesa de la imagen original, en color y en escala de grises
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 5)
        face_gray = gray[y:y+w, x:x+w]
        face_color = frame[y:y+w, x:x+w]


        # Se realiza la detección de ojos en la imagen en gris
        eyes = eye_cascade.detectMultiScale(face_gray, 1.3, 5)

        # Si se encuentran ojos, identificamos las coordenadas
        for (ex, ey, ew, eh) in eyes: 
            cv2.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 5)


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'): break


# Cerramos la captura de imagen y la ventana
cap.release()
cv2.destroyAllWindows()
