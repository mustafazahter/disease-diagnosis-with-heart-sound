"""
Kalp sesi kayıtlarını kullanarak tahmin yapan modül
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import joblib
from tensorflow.keras.models import load_model

from data_loader import HeartSoundDataLoader
from preprocessing import HeartSoundPreprocessor
from feature_extraction import HeartSoundFeatureExtractor
from model import HeartSoundClassifier


def load_and_preprocess_audio(audio_path, target_sr=2000, segment_length_sec=3.0, overlap_ratio=0.5):
    """
    Ses dosyasını yükle ve ön işle
    
    Args:
        audio_path (str): Ses dosyası yolu
        target_sr (int): Hedef örnekleme hızı
        segment_length_sec (float): Segment uzunluğu (saniye)
        overlap_ratio (float): Örtüşme oranı (0-1 arası)
        
    Returns:
        tuple: (işlenmiş segmentler, örnekleme hızı)
    """
    print(f"Ses dosyası yükleniyor: {audio_path}")
    
    # Ses dosyasını yükle
    import librosa
    audio_data, fs = librosa.load(audio_path, sr=None)
    
    print(f"Örnekleme hızı: {fs} Hz")
    print(f"Kayıt uzunluğu: {len(audio_data) / fs:.2f} saniye")
    
    # Önişleyici oluştur
    preprocessor = HeartSoundPreprocessor(target_sr=target_sr)
    
    # Ses verisini işle
    processed_audio = preprocessor.process_audio(audio_data, fs)
    
    # Ses verisini segmentlere ayır
    segments = preprocessor.segment(processed_audio, segment_length_sec, overlap_ratio)
    
    print(f"Segment sayısı: {len(segments)}")
    
    return np.array(segments), fs


def load_from_record(data_dir, record_name, target_sr=2000, segment_length_sec=3.0, overlap_ratio=0.5):
    """
    PhysioNet kayıt adına göre ses dosyasını yükle ve ön işle
    
    Args:
        data_dir (str): Veri klasörü yolu
        record_name (str): Kayıt adı
        target_sr (int): Hedef örnekleme hızı
        segment_length_sec (float): Segment uzunluğu (saniye)
        overlap_ratio (float): Örtüşme oranı (0-1 arası)
        
    Returns:
        tuple: (işlenmiş segmentler, örnekleme hızı)
    """
    print(f"Kayıt yükleniyor: {record_name}")
    
    # Veri yükleyici oluştur
    loader = HeartSoundDataLoader(data_dir)
    
    # Ses verisini yükle
    audio_data, fs = loader.load_audio(record_name)
    
    if audio_data is None:
        raise ValueError(f"Kayıt yüklenemedi: {record_name}")
    
    print(f"Örnekleme hızı: {fs} Hz")
    print(f"Kayıt uzunluğu: {len(audio_data) / fs:.2f} saniye")
    
    # Önişleyici oluştur
    preprocessor = HeartSoundPreprocessor(target_sr=target_sr)
    
    # Ses verisini işle
    processed_audio = preprocessor.process_audio(audio_data, fs)
    
    # Ses verisini segmentlere ayır
    segments = preprocessor.segment(processed_audio, segment_length_sec, overlap_ratio)
    
    print(f"Segment sayısı: {len(segments)}")
    
    return np.array(segments), fs


def prepare_features(segments, feature_type='raw_waveform', feature_extractor_path=None):
    """
    Segmentlerden öznitelik çıkar
    
    Args:
        segments (numpy.ndarray): Ses segmentleri
        feature_type (str): Öznitelik tipi ('raw_waveform', 'features', 'spectrogram')
        feature_extractor_path (str, optional): Öznitelik çıkarıcı model yolu
        
    Returns:
        numpy.ndarray: Öznitelikler
    """
    print(f"Öznitelik çıkarılıyor: {feature_type}")
    
    if feature_type == 'raw_waveform':
        # Ham ses dalgaformunu kullan (CNN modeli için)
        X = segments.reshape(segments.shape[0], segments.shape[1], 1)
    
    elif feature_type == 'features':
        if feature_extractor_path is None:
            # Öznitelik çıkarıcı oluştur
            feature_extractor = HeartSoundFeatureExtractor(target_sr=2000)
        else:
            # Öznitelik çıkarıcıyı yükle
            feature_extractor = joblib.load(feature_extractor_path)
        
        # Her segmentten öznitelik çıkar
        features_df = feature_extractor.extract_features_from_segments(segments)
        
        # Segment_id sütununu çıkar
        feature_cols = [col for col in features_df.columns if col != 'segment_id']
        
        # Öznitelikleri ölçeklendir
        scaled_features_df = feature_extractor.scale_features(features_df[feature_cols], fit=False)
        
        # Çıktıları düzenle
        X = scaled_features_df.values.reshape(scaled_features_df.shape[0], scaled_features_df.shape[1], 1)
    
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
    
    else:
        raise ValueError(f"Bilinmeyen öznitelik tipi: {feature_type}")
    
    return X


def make_prediction(X, model_path):
    """
    Modeli kullanarak tahmin yap
    
    Args:
        X (numpy.ndarray): Öznitelikler
        model_path (str): Model dosyası yolu
        
    Returns:
        tuple: (segment tahminleri, segment olasılıkları, kayıt tahmini, kayıt olasılığı)
    """
    print(f"Model yükleniyor: {model_path}")
    
    # Sınıflandırıcı oluştur
    classifier = HeartSoundClassifier()
    
    # Modeli yükle
    classifier.load_model(model_path)
    
    # Tahmin yap
    print("Tahmin yapılıyor...")
    y_pred, y_pred_prob = classifier.predict(X)
    
    # Segment tahminlerini görselleştir
    segment_predictions = y_pred
    segment_probabilities = y_pred_prob.flatten() if len(y_pred_prob.shape) == 2 else y_pred_prob
    
    # Kayıt tahmini (çoğunluk oylaması)
    record_prediction = 1 if np.mean(segment_predictions) >= 0.5 else 0
    record_probability = np.mean(segment_probabilities)
    
    return segment_predictions, segment_probabilities, record_prediction, record_probability


def visualize_predictions(segments, segment_predictions, segment_probabilities, fs=2000):
    """
    Tahminleri görselleştir
    
    Args:
        segments (numpy.ndarray): Ses segmentleri
        segment_predictions (numpy.ndarray): Segment tahminleri
        segment_probabilities (numpy.ndarray): Segment olasılıkları
        fs (int): Örnekleme hızı
    """
    n_segments = len(segments)
    
    # Segmentleri ve tahminleri görselleştir
    plt.figure(figsize=(15, 10))
    
    for i in range(min(n_segments, 9)):  # En fazla 9 segment göster
        plt.subplot(3, 3, i+1)
        
        # Segment verisi
        t = np.linspace(0, len(segments[i]) / fs, len(segments[i]))
        plt.plot(t, segments[i])
        
        # Segment tahmini
        prediction_text = 'Murmur' if segment_predictions[i] == 1 else 'Normal'
        probability_text = f'{segment_probabilities[i]:.2f}'
        
        plt.title(f"Segment {i+1}: {prediction_text} (P={probability_text})")
        plt.xlabel("Zaman (s)")
        plt.ylabel("Genlik")
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Segment tahminlerini ve olasılıklarını görselleştir
    plt.figure(figsize=(12, 6))
    
    # Tahminler
    plt.subplot(2, 1, 1)
    plt.bar(range(n_segments), segment_predictions)
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.title("Segment Tahminleri")
    plt.xlabel("Segment Indeksi")
    plt.ylabel("Tahmin (0: Normal, 1: Murmur)")
    plt.grid(True)
    
    # Olasılıklar
    plt.subplot(2, 1, 2)
    plt.bar(range(n_segments), segment_probabilities)
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.title("Segment Olasılıkları")
    plt.xlabel("Segment Indeksi")
    plt.ylabel("Murmur Olasılığı")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description='Kalp sesi kayıtlarında murmur tahmin et')
    
    # Giriş parametreleri
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--audio_path', type=str, help='Ses dosyası yolu')
    input_group.add_argument('--record_name', type=str, help='PhysioNet kayıt adı')
    
    # Diğer parametreler
    parser.add_argument('--data_dir', type=str, default='../data',
                      help='Veri klasörü yolu (record_name kullanılırsa)')
    parser.add_argument('--model_path', type=str, default='../models/heart_sound_model.h5',
                      help='Model dosyası yolu')
    parser.add_argument('--feature_extractor_path', type=str, default='../models/feature_extractor.pkl',
                      help='Öznitelik çıkarıcı model yolu')
    parser.add_argument('--feature_type', type=str, default='raw_waveform',
                      choices=['raw_waveform', 'features', 'spectrogram'],
                      help='Öznitelik tipi')
    parser.add_argument('--segment_length', type=float, default=3.0,
                      help='Segment uzunluğu (saniye)')
    parser.add_argument('--overlap_ratio', type=float, default=0.5,
                      help='Segment örtüşme oranı')
    parser.add_argument('--visualize', action='store_true',
                      help='Tahminleri görselleştir')
    
    args = parser.parse_args()
    
    # Ses verisini yükle ve ön işle
    if args.audio_path:
        segments, fs = load_and_preprocess_audio(
            args.audio_path,
            segment_length_sec=args.segment_length,
            overlap_ratio=args.overlap_ratio
        )
    else:
        segments, fs = load_from_record(
            args.data_dir,
            args.record_name,
            segment_length_sec=args.segment_length,
            overlap_ratio=args.overlap_ratio
        )
    
    # Öznitelikleri çıkar
    X = prepare_features(
        segments,
        feature_type=args.feature_type,
        feature_extractor_path=args.feature_extractor_path if args.feature_type == 'features' else None
    )
    
    # Tahmin yap
    segment_predictions, segment_probabilities, record_prediction, record_probability = make_prediction(
        X,
        args.model_path
    )
    
    # Sonuçları yazdır
    print("\nTahmin Sonuçları:")
    print(f"Kayıt tahmini: {'Murmur' if record_prediction == 1 else 'Normal'}")
    print(f"Kayıt olasılığı: {record_probability:.4f}")
    print(f"Murmur segmentleri: {sum(segment_predictions)} / {len(segment_predictions)}")
    
    # Tahminleri görselleştir
    if args.visualize:
        visualize_predictions(segments, segment_predictions, segment_probabilities, fs=2000)
    
    return record_prediction, record_probability


if __name__ == "__main__":
    main() 