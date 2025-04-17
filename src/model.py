"""
Kalp sesi sınıflandırma modeli
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, LSTM, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib


class HeartSoundClassifier:
    """Kalp sesi sınıflandırma modeli"""
    
    def __init__(self, model_type='cnn', input_shape=None, num_classes=2):
        """
        Args:
            model_type (str): Model tipi ('mlp', 'cnn', 'lstm', 'hybrid')
            input_shape (tuple): Giriş verisi şekli
            num_classes (int): Sınıf sayısı
        """
        self.model_type = model_type
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_mlp_model(self):
        """
        MLP (Çok Katmanlı Algılayıcı) modeli oluştur
        
        Returns:
            tensorflow.keras.Model: MLP modeli
        """
        inputs = Input(shape=self.input_shape)
        
        x = Flatten()(inputs)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        if self.num_classes == 2:
            outputs = Dense(1, activation='sigmoid')(x)
        else:
            outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def build_cnn_model(self):
        """
        CNN (Evrişimli Sinir Ağı) modeli oluştur
        
        Returns:
            tensorflow.keras.Model: CNN modeli
        """
        inputs = Input(shape=self.input_shape)
        
        # İlk evrişim bloğu
        x = Conv1D(32, kernel_size=5, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        
        # İkinci evrişim bloğu
        x = Conv1D(64, kernel_size=5, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        
        # Üçüncü evrişim bloğu
        x = Conv1D(128, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.3)(x)
        
        # Düzleştirme ve tam bağlantı katmanları
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        if self.num_classes == 2:
            outputs = Dense(1, activation='sigmoid')(x)
        else:
            outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def build_lstm_model(self):
        """
        LSTM (Uzun-Kısa Süreli Bellek) modeli oluştur
        
        Returns:
            tensorflow.keras.Model: LSTM modeli
        """
        inputs = Input(shape=self.input_shape)
        
        # LSTM katmanları
        x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
        x = Dropout(0.2)(x)
        
        x = Bidirectional(LSTM(32))(x)
        x = Dropout(0.3)(x)
        
        # Tam bağlantı katmanları
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        if self.num_classes == 2:
            outputs = Dense(1, activation='sigmoid')(x)
        else:
            outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def build_hybrid_model(self):
        """
        Hibrit (CNN-LSTM) modeli oluştur
        
        Returns:
            tensorflow.keras.Model: Hibrit model
        """
        inputs = Input(shape=self.input_shape)
        
        # CNN katmanları
        x = Conv1D(32, kernel_size=5, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        x = Conv1D(64, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        # LSTM katmanları
        x = Bidirectional(LSTM(32, return_sequences=False))(x)
        x = Dropout(0.3)(x)
        
        # Tam bağlantı katmanları
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        if self.num_classes == 2:
            outputs = Dense(1, activation='sigmoid')(x)
        else:
            outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def build_model(self):
        """
        Seçilen model tipine göre model oluştur
        
        Returns:
            tensorflow.keras.Model: Seçilen model
        """
        if self.model_type == 'mlp':
            self.model = self.build_mlp_model()
        elif self.model_type == 'cnn':
            self.model = self.build_cnn_model()
        elif self.model_type == 'lstm':
            self.model = self.build_lstm_model()
        elif self.model_type == 'hybrid':
            self.model = self.build_hybrid_model()
        else:
            raise ValueError(f"Bilinmeyen model tipi: {self.model_type}")
        
        # Model derleme
        if self.num_classes == 2:
            self.model.compile(
                loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )
        else:
            self.model.compile(
                loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )
        
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, batch_size=32, epochs=50, 
              model_path='../models/heart_sound_model.h5'):
        """
        Modeli eğit
        
        Args:
            X_train (numpy.ndarray): Eğitim verileri
            y_train (numpy.ndarray): Eğitim etiketleri
            X_val (numpy.ndarray, optional): Doğrulama verileri
            y_val (numpy.ndarray, optional): Doğrulama etiketleri
            batch_size (int): Batch boyutu
            epochs (int): Epoch sayısı
            model_path (str): Model kayıt yolu
            
        Returns:
            tensorflow.keras.callbacks.History: Eğitim geçmişi
        """
        if self.model is None:
            self.build_model()
        
        # Eğer doğrulama seti verilmemişse, eğitim setinden ayır
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42)
        
        # Çok sınıflı sınıflandırma için one-hot encoding
        if self.num_classes > 2:
            y_train = to_categorical(y_train, self.num_classes)
            y_val = to_categorical(y_val, self.num_classes)
        
        # Model kaydetme callback'i
        checkpoint = ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        )
        
        # Erken durdurma callback'i
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        )
        
        # Öğrenme oranı azaltma callback'i
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        callbacks = [checkpoint, early_stopping, reduce_lr]
        
        # Modeli eğit
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Modeli değerlendir
        
        Args:
            X_test (numpy.ndarray): Test verileri
            y_test (numpy.ndarray): Test etiketleri
            
        Returns:
            tuple: (loss, accuracy)
        """
        if self.model is None:
            raise ValueError("Model eğitilmemiş. Önce 'train' metodunu çağırın.")
        
        # Çok sınıflı sınıflandırma için one-hot encoding
        y_test_orig = y_test.copy()
        if self.num_classes > 2:
            y_test = to_categorical(y_test, self.num_classes)
        
        # Modeli değerlendir
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Sınıflandırma metrikleri
        y_pred_prob = self.model.predict(X_test)
        
        if self.num_classes == 2:
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            report = classification_report(y_test_orig, y_pred)
        else:
            y_pred = np.argmax(y_pred_prob, axis=1)
            report = classification_report(y_test_orig, y_pred)
        
        print("\nClassification Report:")
        print(report)
        
        # Karmaşıklık matrisi
        cm = confusion_matrix(y_test_orig, y_pred)
        
        return loss, accuracy
    
    def plot_training_history(self, history):
        """
        Eğitim geçmişini görselleştir
        
        Args:
            history (tensorflow.keras.callbacks.History): Eğitim geçmişi
        """
        plt.figure(figsize=(12, 4))
        
        # Doğruluk grafiği
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Doğruluğu')
        plt.ylabel('Doğruluk')
        plt.xlabel('Epoch')
        plt.legend(['Eğitim', 'Doğrulama'], loc='lower right')
        
        # Kayıp grafiği
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Kaybı')
        plt.ylabel('Kayıp')
        plt.xlabel('Epoch')
        plt.legend(['Eğitim', 'Doğrulama'], loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, X_test, y_test, class_names=None):
        """
        Karmaşıklık matrisini görselleştir
        
        Args:
            X_test (numpy.ndarray): Test verileri
            y_test (numpy.ndarray): Test etiketleri
            class_names (list, optional): Sınıf isimleri
        """
        if self.model is None:
            raise ValueError("Model eğitilmemiş. Önce 'train' metodunu çağırın.")
        
        y_pred_prob = self.model.predict(X_test)
        
        if self.num_classes == 2:
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        else:
            y_pred = np.argmax(y_pred_prob, axis=1)
        
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Karmaşıklık Matrisi')
        plt.colorbar()
        
        if class_names is None:
            if self.num_classes == 2:
                class_names = ['Negatif', 'Pozitif']
            else:
                class_names = [f'Sınıf {i}' for i in range(self.num_classes)]
        
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('Gerçek Etiket')
        plt.xlabel('Tahmin Edilen Etiket')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, X_test, y_test):
        """
        ROC eğrisini görselleştir (ikili sınıflandırma için)
        
        Args:
            X_test (numpy.ndarray): Test verileri
            y_test (numpy.ndarray): Test etiketleri
        """
        if self.num_classes != 2:
            print("ROC eğrisi yalnızca ikili sınıflandırma için kullanılabilir.")
            return
        
        if self.model is None:
            raise ValueError("Model eğitilmemiş. Önce 'train' metodunu çağırın.")
        
        y_pred_prob = self.model.predict(X_test).flatten()
        
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC eğrisi (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Yanlış Pozitif Oranı')
        plt.ylabel('Gerçek Pozitif Oranı')
        plt.title('Alıcı İşletim Karakteristiği (ROC) Eğrisi')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    
    def save_model(self, model_path='../models/heart_sound_model.h5', 
                  scaler_path='../models/scaler.pkl'):
        """
        Modeli kaydet
        
        Args:
            model_path (str): Model kayıt yolu
            scaler_path (str): Ölçekleyici kayıt yolu
        """
        if self.model is None:
            raise ValueError("Model eğitilmemiş. Önce 'train' metodunu çağırın.")
        
        self.model.save(model_path)
        print(f"Model kaydedildi: {model_path}")
    
    def load_model(self, model_path='../models/heart_sound_model.h5'):
        """
        Modeli yükle
        
        Args:
            model_path (str): Model dosyası yolu
        """
        self.model = keras.models.load_model(model_path)
        print(f"Model yüklendi: {model_path}")
    
    def predict(self, X):
        """
        Tahmin yap
        
        Args:
            X (numpy.ndarray): Tahmin edilecek veriler
            
        Returns:
            numpy.ndarray: Tahminler
        """
        if self.model is None:
            raise ValueError("Model eğitilmemiş. Önce 'train' metodunu çağırın veya bir model yükleyin.")
        
        y_pred_prob = self.model.predict(X)
        
        if self.num_classes == 2:
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        else:
            y_pred = np.argmax(y_pred_prob, axis=1)
        
        return y_pred, y_pred_prob


# Test için örnek kullanım
if __name__ == "__main__":
    # Yapay veri oluştur
    # Gerçek veri için öznitelik çıkarma modülünü kullan
    X = np.random.randn(100, 20, 1)  # 100 örnek, 20 zaman adımı, 1 öznitelik
    y = np.random.randint(0, 2, size=100)  # İkili sınıflandırma için rastgele etiketler
    
    # Eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Sınıflandırıcı oluştur
    classifier = HeartSoundClassifier(
        model_type='cnn',
        input_shape=(20, 1),
        num_classes=2
    )
    
    # Modeli oluştur
    model = classifier.build_model()
    model.summary()
    
    # Modeli eğit
    history = classifier.train(
        X_train, y_train,
        batch_size=32,
        epochs=5,  # Test için az sayıda epoch
        model_path='../models/test_model.h5'
    )
    
    # Modeli değerlendir
    classifier.evaluate(X_test, y_test)
    
    # Eğitim geçmişini görselleştir
    classifier.plot_training_history(history)
    
    # Karmaşıklık matrisini görselleştir
    classifier.plot_confusion_matrix(X_test, y_test, class_names=['Normal', 'Murmur'])
    
    # ROC eğrisini görselleştir
    classifier.plot_roc_curve(X_test, y_test)