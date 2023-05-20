import cv2 as cv

face_cascade = cv.CascadeClassifier('C:/Users/saibo/AppData/Roaming/Python/Python39/site-packages/cv2/data/haarcascade_frontalface_default.xml')

filtro = cv.imread("simio.png", cv.IMREAD_UNCHANGED)
VideoCap = cv.VideoCapture(0)

if not VideoCap.isOpened():
    print('No se pudo acceder a la camara')
else:
    while True:
        ret, frame = VideoCap.read()
        frame = cv.flip(frame,1)
        imagenGrises = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)
        faces = face_cascade.detectMultiScale(
            imagenGrises,
            scaleFactor=1.5,
            minNeighbors=5,
            minSize=(30, 30)
        )
        for (x, y, w, h) in faces:
            # Escalar el filtro al tamaÃ±o de la cara detectada
            filtro_resized = cv.resize(filtro, (w, h))

            # Colocar el filtro en la cara detectada
            for i in range(filtro_resized.shape[0]):
                for j in range(filtro_resized.shape[1]):
                    if filtro_resized[i, j][3] != 0:
                        frame[y + i, x + j] = filtro_resized[i, j]

        frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)
        cv.imshow('Video', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
VideoCap.release()
cv.destroyAllWindows()