import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ruta de tu carpeta
dataset_path = "dataset"

# Preparar imágenes
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=2,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=2,
    class_mode='binary',
    subset='validation'
)

# Modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(224,224,3)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entrenar
model.fit(train_data, epochs=3, validation_data=val_data)

# Guardar
model.save("modelo_celulas.h5")

