import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Etiketleri tamsayıya dönüştürmek için bir sözlük oluştur
label_dict = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}

def load_dataset(directory):
    images = []
    labels = []
    for emotion_label, label in label_dict.items():
        emotion_path = os.path.join(directory, emotion_label)
        for filename in os.listdir(emotion_path):
            image_path = os.path.join(emotion_path, filename)
            # Görüntüyü yükleme ve işleme
            image = Image.open(image_path).convert('L').resize((48, 48))
            image_np = np.array(image) / 255.0
            images.append(image_np)
            labels.append(label)
    return np.array(images), np.array(labels)

# Veri setini yükleme
train_images, train_labels = load_dataset('C:/Users/emirh/OneDrive/Masaüstü/proje1/fer-2013/train')
test_images, test_labels = load_dataset('C:/Users/emirh/OneDrive/Masaüstü/proje1/fer-2013/test')

# Eğitim ve test setlerini oluşturma
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# CNN modelini oluşturma
num_classes = 7
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Modeli derleme
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)

# Modelin performansını değerlendirme
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test kaybı:", test_loss)
print("Test doğruluğu:", test_accuracy)

# Eğitim sırasında oluşan kayıp ve doğruluk değerlerini alın
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Kayıp grafiğini oluşturun
plt.figure(figsize=(10, 5))
plt.plot(train_loss, label='Eğitim Kaybı')
plt.plot(val_loss, label='Doğrulama Kaybı')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.xlabel('Epok')
plt.ylabel('Kayıp')
plt.legend()
plt.show()

# Doğruluk grafiğini oluşturun
plt.figure(figsize=(10, 5))
plt.plot(train_acc, label='Eğitim Doğruluğu')
plt.plot(val_acc, label='Doğrulama Doğruluğu')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.xlabel('Epok')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()

def predict_emotion(image_path):
    # Görüntüyü yükleme ve işleme
    image_pil = Image.open(image_path)
    image_pil = image_pil.resize((48, 48))
    image_gray = image_pil.convert('L')
    image_np = np.array(image_gray) / 255.0
    image_np = np.expand_dims(image_np, axis=-1)  # Boyut ekleyerek (48, 48, 1) şekline getirme
    
    # Tahmin yapma
    prediction = model.predict(np.array([image_np]))
    predicted_class = np.argmax(prediction)
    class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]  # Sınıf adları
    emotion = class_names[predicted_class]
    
    return emotion

# Test edilecek görüntü yolu
test_image_path = r'C:\Users\emirh\OneDrive\Masaüstü\proje1\fer-2013\2.jpg'

# Görüntünün hangi duyguda olduğunu tahmin etme
predicted_emotion = predict_emotion(test_image_path)
print("Tahmin edilen duygu:", predicted_emotion)
    
    