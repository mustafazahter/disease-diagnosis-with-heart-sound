"""
Kalp sesi sınıflandırma modelini eğiten modül
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
import joblib

from data_loader import HeartSoundDataLoader
from preprocessing import HeartSoundPreprocessor
from feature_extraction import HeartSoundFeatureExtractor
from model import HeartSoundClassifier


def preprocess_data(data_dir, segment_length_sec=3.0, overlap_ratio=0.5, target_sr=2000):
    """
    Veri setini ön işleme adımlarından geçir
    
    Args:
        data_dir (str): Veri klasörü yolu
        segment_length_sec (float): Segment uzunluğu (saniye)
        overlap_ratio (float): Örtüşme oranı (0-1 arası)
        target_sr (int): Hedef örnekleme hızı
        
    Returns:
        tuple: (işlenmiş segmentler, murmur etiketleri, kayıt isimleri)
    """
    print("Veri yükleniyor ve önişleniyor...")
    
    # Veri yükleyici oluştur
    loader = HeartSoundDataLoader(data_dir)
    
    # Metadata ve kayıt listesini yükle
    metadata = loader.load_metadata()
    records = loader.load_records_list()
    
    # Önişleyici oluştur
    preprocessor = HeartSoundPreprocessor(target_sr=target_sr)
    
    all_segments = []
    all_labels = []
    segment_record_map = []  # Her segmentin hangi kayda ait olduğunu tut
    
    # Her kayıt için
    for record_name in records:
        # Metadata'dan murmur bilgisini al
        record_id = record_name.split('_')[0]  # Kayıt ID'sini çıkar
        
        # Metadata'da kayıt varsa etiketini al
        if record_id in metadata['patient_id'].values:
            record_meta = metadata[metadata['patient_id'] == record_id].iloc[0]
            murmur_label = 1 if record_meta['murmur'] == 'Present' else 0
            
            # Ses verisini yükle
            audio_data, fs = loader.load_audio(record_name)
            
            if audio_data is not None:
                # Ses verisini işle
                processed_audio = preprocessor.process_audio(audio_data, fs)
                
                # Ses verisini segmentlere ayır
                segments = preprocessor.segment(processed_audio, segment_length_sec, overlap_ratio)
                
                # Segmentleri ve etiketleri depola
                for segment in segments:
                    all_segments.append(segment)
                    all_labels.append(murmur_label)
                    segment_record_map.append(record_name)
                
                print(f"Kayıt işlendi: {record_name}, Murmur: {'Var' if murmur_label == 1 else 'Yok'}, Segment sayısı: {len(segments)}")
    
    print(f"Toplam segment sayısı: {len(all_segments)}")
    print(f"Pozitif segment sayısı: {sum(all_labels)}")
    print(f"Negatif segment sayısı: {len(all_labels) - sum(all_labels)}")
    
    return np.array(all_segments), np.array(all_labels), segment_record_map


def extract_features(segments, labels, segment_record_map, feature_type='raw_waveform'):
    """
    Segmentlerden öznitelik çıkar
    
    Args:
        segments (numpy.ndarray): Ses segmentleri
        labels (numpy.ndarray): Etiketler
        segment_record_map (list): Her segmentin hangi kayda ait olduğu
        feature_type (str): Öznitelik tipi ('raw_waveform', 'features', 'spectrogram')
        
    Returns:
        tuple: (öznitelikler, etiketler)
    """
    print(f"Öznitelik çıkarılıyor: {feature_type}")
    
    if feature_type == 'raw_waveform':
        # Ham ses dalgaformunu kullan (CNN modeli için)
        X = segments.reshape(segments.shape[0], segments.shape[1], 1)
        y = labels
    
    elif feature_type == 'features':
        # Öznitelik çıkarıcı oluştur
        feature_extractor = HeartSoundFeatureExtractor(target_sr=2000)
        
        # Her segmentten öznitelik çıkar
        features_df = feature_extractor.extract_features_from_segments(segments)
        
        # Segment_id sütununu çıkar
        feature_cols = [col for col in features_df.columns if col != 'segment_id']
        
        # Öznitelikleri ölçeklendir
        scaled_features_df = feature_extractor.scale_features(features_df[feature_cols])
        
        # Çıktıları düzenle
        X = scaled_features_df.values.reshape(scaled_features_df.shape[0], scaled_features_df.shape[1], 1)
        y = labels
        
        # Öznitelik çıkarıcıyı kaydet
        joblib.dump(feature_extractor, '../models/feature_extractor.pkl')
    
    elif feature_type == 'spectrogram':
        # Spektrogramları hesapla
        import librosa
        
        specs = []
        for segment in segments:
            # Mel spektrogramını hesapla
            mel_spec = librosa.feature.melspectrogram(y=segment, sr=2000, n_mels=128, fmax=1000)
            # dB ölçeğine dönüştür
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            specs.append(mel_spec_db)
        
        # Spektrogramları numpy dizisine dönüştür
        X = np.array(specs)
        y = labels
    
    else:
        raise ValueError(f"Bilinmeyen öznitelik tipi: {feature_type}")
    
    return X, y


def prepare_data_splits(X, y, segment_record_map, test_size=0.2, val_size=0.2):
    """
    Eğitim, doğrulama ve test setlerini hazırla
    
    Args:
        X (numpy.ndarray): Öznitelikler
        y (numpy.ndarray): Etiketler
        segment_record_map (list): Her segmentin hangi kayda ait olduğu
        test_size (float): Test seti oranı
        val_size (float): Doğrulama seti oranı
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("Veri setleri hazırlanıyor...")
    
    # Benzersiz kayıt isimlerini al
    unique_records = np.unique(segment_record_map)
    
    # Kayıtları rastgele karıştır
    np.random.shuffle(unique_records)
    
    # Test ve doğrulama setleri için kayıt sayılarını hesapla
    test_count = int(len(unique_records) * test_size)
    val_count = int(len(unique_records) * val_size)
    
    # Kayıtları setlere ayır
    test_records = unique_records[:test_count]
    val_records = unique_records[test_count:test_count+val_count]
    train_records = unique_records[test_count+val_count:]
    
    print(f"Eğitim kayıt sayısı: {len(train_records)}")
    print(f"Doğrulama kayıt sayısı: {len(val_records)}")
    print(f"Test kayıt sayısı: {len(test_records)}")
    
    # Segment indekslerini setlere ayır
    train_indices = [i for i, rec in enumerate(segment_record_map) if rec in train_records]
    val_indices = [i for i, rec in enumerate(segment_record_map) if rec in val_records]
    test_indices = [i for i, rec in enumerate(segment_record_map) if rec in test_records]
    
    # Veri setlerini oluştur
    X_train = X[train_indices]
    y_train = y[train_indices]
    
    X_val = X[val_indices]
    y_val = y[val_indices]
    
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    print(f"Eğitim segment sayısı: {len(X_train)}")
    print(f"Doğrulama segment sayısı: {len(X_val)}")
    print(f"Test segment sayısı: {len(X_test)}")
    
    # Sınıf dağılımlarını yazdır
    print(f"Eğitim seti sınıf dağılımı: Pozitif={sum(y_train)}, Negatif={len(y_train)-sum(y_train)}")
    print(f"Doğrulama seti sınıf dağılımı: Pozitif={sum(y_val)}, Negatif={len(y_val)-sum(y_val)}")
    print(f"Test seti sınıf dağılımı: Pozitif={sum(y_test)}, Negatif={len(y_test)-sum(y_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train, y_train, X_val, y_val, model_type='cnn', feature_type='raw_waveform',
               batch_size=32, epochs=50, model_path='../models/heart_sound_model.h5'):
    """
    Modeli eğit
    
    Args:
        X_train (numpy.ndarray): Eğitim verileri
        y_train (numpy.ndarray): Eğitim etiketleri
        X_val (numpy.ndarray): Doğrulama verileri
        y_val (numpy.ndarray): Doğrulama etiketleri
        model_type (str): Model tipi ('mlp', 'cnn', 'lstm', 'hybrid')
        feature_type (str): Öznitelik tipi ('raw_waveform', 'features', 'spectrogram')
        batch_size (int): Batch boyutu
        epochs (int): Epoch sayısı
        model_path (str): Model kayıt yolu
        
    Returns:
        tuple: (model, history)
    """
    print(f"Model eğitiliyor: {model_type}")
    
    # Giriş şeklini belirle
    if feature_type == 'raw_waveform':
        input_shape = (X_train.shape[1], 1)  # (zaman adımları, kanal sayısı)
    elif feature_type == 'features':
        input_shape = (X_train.shape[1], 1)  # (öznitelik sayısı, kanal sayısı)
    elif feature_type == 'spectrogram':
        input_shape = X_train.shape[1:]  # (frekans bantları, zaman adımları)
    
    # Sınıflandırıcı oluştur
    classifier = HeartSoundClassifier(
        model_type=model_type,
        input_shape=input_shape,
        num_classes=2
    )
    
    # Modeli oluştur
    model = classifier.build_model()
    model.summary()
    
    # Sınıf ağırlıklarını hesapla
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    # Modeli eğit
    history = classifier.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        batch_size=batch_size,
        epochs=epochs,
        model_path=model_path
    )
    
    # Modeli kaydet
    classifier.save_model(model_path)
    
    return classifier, history


def evaluate_model(classifier, X_test, y_test, class_names=['Normal', 'Murmur']):
    """
    Modeli değerlendir
    
    Args:
        classifier (HeartSoundClassifier): Sınıflandırıcı
        X_test (numpy.ndarray): Test verileri
        y_test (numpy.ndarray): Test etiketleri
        class_names (list): Sınıf isimleri
    """
    print("Model değerlendiriliyor...")
    
    # Test seti üzerinde değerlendir
    loss, accuracy = classifier.evaluate(X_test, y_test)
    
    # Karmaşıklık matrisini görselleştir
    classifier.plot_confusion_matrix(X_test, y_test, class_names=class_names)
    
    # ROC eğrisini görselleştir
    classifier.plot_roc_curve(X_test, y_test)
    
    return loss, accuracy


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description='Kalp sesi sınıflandırma modeli eğitimi')
    
    parser.add_argument('--data_dir', type=str, default='../data',
                      help='Veri klasörü yolu')
    parser.add_argument('--model_type', type=str, default='cnn',
                      choices=['mlp', 'cnn', 'lstm', 'hybrid'],
                      help='Model tipi')
    parser.add_argument('--feature_type', type=str, default='raw_waveform',
                      choices=['raw_waveform', 'features', 'spectrogram'],
                      help='Öznitelik tipi')
    parser.add_argument('--segment_length', type=float, default=3.0,
                      help='Segment uzunluğu (saniye)')
    parser.add_argument('--overlap_ratio', type=float, default=0.5,
                      help='Segment örtüşme oranı')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch boyutu')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Epoch sayısı')
    parser.add_argument('--model_path', type=str, default='../models/heart_sound_model.h5',
                      help='Model kayıt yolu')
    
    args = parser.parse_args()
    
    # Veriyi ön işle
    segments, labels, segment_record_map = preprocess_data(
        args.data_dir,
        segment_length_sec=args.segment_length,
        overlap_ratio=args.overlap_ratio
    )
    
    # Öznitelikleri çıkar
    X, y = extract_features(
        segments,
        labels,
        segment_record_map,
        feature_type=args.feature_type
    )
    
    # Veri setlerini hazırla
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(
        X, y, segment_record_map
    )
    
    # Modeli eğit
    classifier, history = train_model(
        X_train, y_train,
        X_val, y_val,
        model_type=args.model_type,
        feature_type=args.feature_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        model_path=args.model_path
    )
    
    # Eğitim geçmişini görselleştir
    classifier.plot_training_history(history)
    
    # Modeli değerlendir
    evaluate_model(classifier, X_test, y_test)


if __name__ == "__main__":
    main() 