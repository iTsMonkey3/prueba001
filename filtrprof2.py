# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:48:34 2019

@author: GabrielAsus
"""
import cv2 as cv
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv.CascadeClassifier(cascPath)

#obtener acceso a la webcam
video_capture = cv.VideoCapture(0,cv.CAP_DSHOW)
#video_capture = cv.VideoCapture('rtsp://192.168.8.5:5554/front')
anterior = 0
if not video_capture.isOpened():
        print('No se pudo acceder a la camara')
else:
    while True:
        #revisar si ya puedo leer imagenes de la camara
        ret, frame = video_capture.read()
        frame=cv.flip(frame,1)
        imagenGrises = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
        faces = faceCascade.detectMultiScale(
            imagenGrises,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        #por cada cara detectada pintar un cuadro
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 250, 0), -1)
        # Mostrar la deteccion
        cv.imshow('Video', frame)
        #se motraran las caras mientra no presionemos la tecla q
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    #liberar la camara
    video_capture.release()
    #cerrar todas las ventanas
    cv.destroyAllWindows()