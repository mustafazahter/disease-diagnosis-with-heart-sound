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
    
    # Metadata ya da kayıt listesi boşsa hata ver
    if metadata is None or records is None:
        print("HATA: Metadata veya kayıt listesi yüklenemedi!")
        return np.array([]), np.array([]), []
    
    # Metadata sütunlarını kontrol et
    required_columns = ['Patient ID', 'Murmur']
    for col in required_columns:
        if col not in metadata.columns:
            print(f"HATA: Metadata dosyasında gerekli sütun bulunamadı: {col}")
            print(f"Mevcut sütunlar: {metadata.columns.tolist()}")
            return np.array([]), np.array([]), []
    
    # Önişleyici oluştur
    preprocessor = HeartSoundPreprocessor(target_sr=target_sr)
    
    all_segments = []
    all_labels = []
    segment_record_map = []  # Her segmentin hangi kayda ait olduğunu tut
    
    # Eşleştirme istatistikleri
    matched_count = 0
    unmatched_count = 0
    failed_audio_count = 0
    success_audio_count = 0
    
    print(f"Toplam kayıt sayısı: {len(records)}")
    print(f"İlk 5 kayıt: {records[:5]}")
    print(f"İlk 5 hasta ID: {metadata['Patient ID'].values[:5]}")
    
    # Her kayıt için
    for record_name in records:
        try:
            # Metadata'dan murmur bilgisini al
            # Kayıt ID'sini doğru şekilde çıkar
            # Örnek: 'training_data/2530_AV' -> '2530'
            if '/' in record_name:
                # 'training_data/2530_AV' formatındaysa
                record_id = record_name.split('/')[-1].split('_')[0]
            else:
                # '2530_AV' formatındaysa
                record_id = record_name.split('_')[0]
            
            # Debug bilgisi
            if matched_count < 5 and unmatched_count < 5:
                print(f"Kayıt adı: {record_name}, Çıkarılan ID: {record_id}")
            
            # Metadata'da kayıt varsa etiketini al
            if record_id in metadata['Patient ID'].values.astype(str):
                record_meta = metadata[metadata['Patient ID'].astype(str) == record_id].iloc[0]
                murmur_label = 1 if record_meta['Murmur'] == 'Present' else 0
                
                # Ses verisini yükle
                audio_data, fs = loader.load_audio(record_name)
                
                if audio_data is not None:
                    try:
                        # NaN veya Infinity değerleri kontrolü
                        if np.isnan(audio_data).any() or np.isinf(audio_data).any():
                            print(f"UYARI: {record_name} kaydında NaN veya Infinity değerleri var. Temizleniyor...")
                            audio_data = np.nan_to_num(audio_data)
                            
                        # Ses verisini işle
                        processed_audio = preprocessor.process_audio(audio_data, fs)
                        
                        # Tekrar NaN veya Infinity kontrolü
                        if np.isnan(processed_audio).any() or np.isinf(processed_audio).any():
                            print(f"UYARI: {record_name} kaydı işlendikten sonra hala NaN veya Infinity değerleri var. Bu kayıt atlanıyor.")
                            failed_audio_count += 1
                            continue
                        
                        # Ses verisini segmentlere ayır
                        segments = preprocessor.segment(processed_audio, segment_length_sec, overlap_ratio)
                        
                        # Son kontrol: segmentlerde NaN veya Infinity var mı?
                        valid_segments = []
                        for segment in segments:
                            if not np.isnan(segment).any() and not np.isinf(segment).any():
                                valid_segments.append(segment)
                            else:
                                print(f"UYARI: {record_name} kaydının bir segmentinde NaN veya Infinity değerleri var. Bu segment atlanıyor.")
                        
                        if valid_segments:
                            # Segmentleri ve etiketleri depola
                            for segment in valid_segments:
                                all_segments.append(segment)
                                all_labels.append(murmur_label)
                                segment_record_map.append(record_name)
                            
                            print(f"Kayıt işlendi: {record_name}, Murmur: {'Var' if murmur_label == 1 else 'Yok'}, Segment sayısı: {len(valid_segments)}")
                            matched_count += 1
                            success_audio_count += 1
                        else:
                            print(f"UYARI: {record_name} kaydından geçerli segment oluşturulamadı.")
                            failed_audio_count += 1
                    except Exception as e:
                        print(f"HATA: Kayıt işlenirken bir sorun oluştu: {record_name} - {str(e)}")
                        failed_audio_count += 1
                else:
                    print(f"UYARI: Ses verisi yüklenemedi: {record_name}")
                    failed_audio_count += 1
            else:
                unmatched_count += 1
                if unmatched_count <= 5:  # Sadece ilk 5 eşleşmeyen kaydı göster
                    print(f"UYARI: Kayıt ID'si metadata'da bulunamadı: {record_id} ({record_name})")
        except Exception as e:
            print(f"HATA: Kayıt işlenirken bir sorun oluştu: {record_name} - {str(e)}")
            failed_audio_count += 1
    
    # İstatistikleri görüntüle
    print("\nEşleştirme İstatistikleri:")
    print(f"Toplam kayıt sayısı: {len(records)}")
    print(f"Metadata ile eşleşen kayıt sayısı: {matched_count}")
    print(f"Metadata ile eşleşmeyen kayıt sayısı: {unmatched_count}")
    print(f"Ses verisi yüklenip işlenen kayıt sayısı: {success_audio_count}")
    print(f"Ses verisi yüklenemeyen veya işlenemeyen kayıt sayısı: {failed_audio_count}")
    
    print(f"\nToplam segment sayısı: {len(all_segments)}")
    print(f"Pozitif segment sayısı: {sum(all_labels)}")
    print(f"Negatif segment sayısı: {len(all_labels) - sum(all_labels)}")
    
    if len(all_segments) == 0:
        print("\nDİKKAT: Hiç segment oluşturulamadı. Lütfen kontrol edin:")
        print("1. Kayıt ID'leri ile metadata'daki ID'ler eşleşiyor mu?")
        print("2. Ses dosyaları (.hea ve .dat) doğru konumda mı?")
        print("3. Metadata dosyasındaki sütun isimleri ve değerleri doğru formatta mı?")
    
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
    
    # Eğer segment listesi boşsa, hata vermeden boş diziler döndür
    if len(segments) == 0:
        print("UYARI: Hiç segment bulunamadı. Veri yükleme ve önişleme adımlarını kontrol edin.")
        return np.array([]), np.array([])
    
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
    
    parser.add_argument('--data_dir', type=str, 
                      default='/content/disease-diagnosis-with-heart-sound/data/the-circor-digiscope-phonocardiogram-dataset-1.0.3/',
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
    
    # Segment sayısı kontrolü
    if len(segments) == 0:
        print("HATA: Hiç ses segmenti bulunamadı. İşlem durduruluyor.")
        print("Olası sebepler:")
        print("1. Ses dosyaları (.hea ve .dat) bulunamıyor olabilir")
        print("2. Kayıt ID'leri ile metadata arasında eşleşme sorunu olabilir")
        print("3. Metadata dosyasındaki sütun isimleri beklenen formatta olmayabilir")
        print("\nLütfen aşağıdaki adımları kontrol edin:")
        print("- Ses dosyaları ve metadata doğru konumda mı?")
        print("- RECORDS dosyası doğru kayıt ID'lerini içeriyor mu?")
        print("- data_loader.py dosyasını önce tek başına çalıştırıp veri yüklemeyi test edin")
        return
    
    # Öznitelikleri çıkar
    X, y = extract_features(
        segments,
        labels,
        segment_record_map,
        feature_type=args.feature_type
    )
    
    # Öznitelik sayısı kontrolü
    if len(X) == 0:
        print("HATA: Öznitelik çıkarma işlemi başarısız oldu. İşlem durduruluyor.")
        return
    
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