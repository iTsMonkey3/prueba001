# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:48:34 2019

@author: GabrielAsus
"""
import cv2 as cv
cascPath = "C:/Users/saibo/AppData/Roaming/Python/Python39/site-packages/cv2/data/haarcascade_frontalface_default.xml"
faceCascade = cv.CascadeClassifier(cascPath)
elMono=cv.imread("nacho2.png",cv.IMREAD_UNCHANGED)

#obtener acceso a la webcam
video_capture = cv.VideoCapture(0,cv.CAP_DSHOW)

if not video_capture.isOpened():
        print('No se pudo acceder a la camara')
else:
    while True:
        #revisar si ya puedo leer imagenes de la camara
        ret, frame = video_capture.read()
        frame=cv.flip(frame,1)
        imagenGrises = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)
        faces = faceCascade.detectMultiScale(
            imagenGrises,
            scaleFactor=1.5,
            minNeighbors=5,
            minSize=(30, 30)
        )
        for (x, y, w, h) in faces:
            filtro_resized = cv.resize(elMono, (w, h))
        
        # Poner filtro
            for i in range(filtro_resized.shape[0]):
                for j in range(filtro_resized.shape[1]):
                    if filtro_resized[i, j][3] != 0:
                        frame[y + i, x + j] = filtro_resized[i, j]
        # Mostrar la deteccion
        cv.imshow('Video', frame)
        #se motraran las caras mientra no presionemos la tecla q
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    #liberar la camara
    video_capture.release()
    #cerrar todas las ventanas
    cv.destroyAllWindows()