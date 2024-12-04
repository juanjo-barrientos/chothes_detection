import cv2
import numpy as np
import tensorflow as tf

# Cargar el modelo preentrenado de Fashion MNIST (cambia esto por la ruta de tu computadora)
model = tf.keras.models.load_model('C:/Users/JUANJO/Documents/data_science/clothes_detector/model/second_version_cnn.h5')

# Etiquetas de Fashion MNIST
labels = ['Camiseta', 'pantalon', 'Jersey', 'Vestido', 'Abrigo', 
          'Sandalia', 'Camisa', 'Tenis', 'Bolso', 'Bota']

# Inicializar la captura de video con la cámara del portátil
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abre correctamente
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

while True:
    # Leer frame desde la cámara
    ret, frame = cap.read()
    
    if not ret:
        print("No se pudo recibir el frame. Saliendo...")
        break
    
    # Redimensionar la imagen capturada al tamaño esperado por MobileNetV2 (224x224)
    img_resized = cv2.resize(frame, (224, 224))
    
    # Preprocesar la imagen: convertirla a array, cambiar formato a RGB, expandir dimensiones, aplicar normalización
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_array = np.array(img_rgb, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)  # Preprocesamiento específico de MobileNetV2
    
    # Realizar la predicción
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Obtener la etiqueta correspondiente
    label = labels[predicted_class]
    
    # Mostrar la clase predicha en la ventana de video
    cv2.putText(frame, f'Prenda detectada: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Mostrar el frame con la etiqueta de predicción
    cv2.imshow('Detección de prendas', frame)
    
    # Presionar 'q' para salir del loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
