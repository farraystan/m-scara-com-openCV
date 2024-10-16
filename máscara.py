import cv2

# Carregar o classificador de rosto pré-treinado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Carregar a imagem da máscara
mask = cv2.imread('coraline.png', cv2.IMREAD_UNCHANGED)

# Iniciar a captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converter o quadro para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostos no quadro
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Redimensionar a máscara para o tamanho do rosto
        mask_resized = cv2.resize(mask, (w, h))

        # Verificar se a máscara tem um canal alfa
        if mask_resized.shape[2] == 4:
            mask_rgb = mask_resized[:, :, :3]
            mask_alpha = mask_resized[:, :, 3]

            # Selecionar a região do rosto no quadro original
            roi = frame[y:y+h, x:x+w]

            # Combinar a máscara com a região do rosto
            for c in range(3):
                roi[:, :, c] = roi[:, :, c] * (1 - mask_alpha / 255.0) + mask_rgb[:, :, c] * (mask_alpha / 255.0)
        else:
            mask_rgb = mask_resized

            # Selecionar a região do rosto no quadro original
            roi = frame[y:y+h, x:x+w]

            # Combinar a máscara com a região do rosto
            for c in range(3):
                roi[:, :, c] = mask_rgb[:, :, c]

    # Exibir o quadro resultante
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura e fechar as janelas
cap.release()
cv2.destroyAllWindows()
