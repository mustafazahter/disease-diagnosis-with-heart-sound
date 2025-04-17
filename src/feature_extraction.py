"""
Kalp sesi kayıtlarından öznitelik çıkarma modülü
"""

import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from sklearn.preprocessing import StandardScaler


class HeartSoundFeatureExtractor:
    """Kalp sesi öznitelik çıkarma sınıfı"""
    
    def __init__(self, target_sr=2000):
        """
        Args:
            target_sr (int): Hedef örnekleme hızı
        """
        self.target_sr = target_sr
        self.scaler = StandardScaler()
    
    def extract_time_domain_features(self, audio):
        """
        Zaman domaininde öznitelik çıkar
        
        Args:
            audio (numpy.ndarray): Ses verisi
            
        Returns:
            dict: Zaman domaini özellikleri
        """
        features = {}
        
        # İstatistiksel özellikler
        features['mean'] = np.mean(audio)
        features['std'] = np.std(audio)
        features['max'] = np.max(audio)
        features['min'] = np.min(audio)
        features['range'] = np.max(audio) - np.min(audio)
        features['median'] = np.median(audio)
        features['q1'] = np.percentile(audio, 25)
        features['q3'] = np.percentile(audio, 75)
        features['iqr'] = features['q3'] - features['q1']
        features['rms'] = np.sqrt(np.mean(np.square(audio)))
        
        # İçerik şekil özellikleri
        features['skewness'] = stats.skew(audio)
        features['kurtosis'] = stats.kurtosis(audio)
        features['crest_factor'] = np.max(np.abs(audio)) / features['rms'] if features['rms'] > 0 else 0
        
        # Sıfır geçiş oranı
        features['zero_crossing_rate'] = np.sum(np.abs(np.diff(np.signbit(audio)))) / len(audio)
        
        return features
    
    def extract_frequency_domain_features(self, audio):
        """
        Frekans domaininde öznitelik çıkar
        
        Args:
            audio (numpy.ndarray): Ses verisi
            
        Returns:
            dict: Frekans domaini özellikleri
        """
        features = {}
        
        # FFT (Hızlı Fourier Dönüşümü)
        fft = np.abs(np.fft.rfft(audio))
        freq = np.fft.rfftfreq(len(audio), 1/self.target_sr)
        
        # Frekans domaini özellikleri
        features['spectral_centroid'] = np.sum(freq * fft) / np.sum(fft) if np.sum(fft) > 0 else 0
        features['spectral_mean'] = np.mean(fft)
        features['spectral_std'] = np.std(fft)
        features['spectral_skew'] = stats.skew(fft)
        features['spectral_kurtosis'] = stats.kurtosis(fft)
        
        # Spektral düşüş
        cumsum = np.cumsum(fft)
        spectral_rolloff_point = np.searchsorted(cumsum, 0.85 * cumsum[-1])
        features['spectral_rolloff'] = freq[spectral_rolloff_point] if spectral_rolloff_point < len(freq) else 0
        
        # Spektral düzlük
        geometric_mean = np.exp(np.mean(np.log(fft + 1e-10)))
        arithmetic_mean = np.mean(fft)
        features['spectral_flatness'] = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
        
        # Bant enerjileri
        bands = [
            (25, 45),    # Düşük frekans bandı
            (45, 80),    # S1 bandı
            (80, 200),   # Orta frekans bandı
            (200, 400)   # Yüksek frekans bandı
        ]
        
        for i, (low, high) in enumerate(bands):
            low_idx = np.searchsorted(freq, low)
            high_idx = np.searchsorted(freq, high)
            band_energy = np.sum(fft[low_idx:high_idx])
            total_energy = np.sum(fft)
            features[f'band_energy_{low}_{high}'] = band_energy
            features[f'band_energy_ratio_{low}_{high}'] = band_energy / total_energy if total_energy > 0 else 0
        
        return features
    
    def extract_mfcc_features(self, audio, n_mfcc=13):
        """
        MFCC özniteliklerini çıkar
        
        Args:
            audio (numpy.ndarray): Ses verisi
            n_mfcc (int): MFCC sayısı
            
        Returns:
            dict: MFCC özellikleri
        """
        features = {}
        
        # MFCC hesapla
        mfccs = librosa.feature.mfcc(y=audio, sr=self.target_sr, n_mfcc=n_mfcc)
        
        # Her bir MFCC için ortalama ve standart sapma
        for i in range(n_mfcc):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
        
        # Delta (1. türev) MFCC
        delta_mfccs = librosa.feature.delta(mfccs)
        
        for i in range(n_mfcc):
            features[f'delta_mfcc_{i+1}_mean'] = np.mean(delta_mfccs[i])
            features[f'delta_mfcc_{i+1}_std'] = np.std(delta_mfccs[i])
        
        # Delta-delta (2. türev) MFCC
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        for i in range(n_mfcc):
            features[f'delta2_mfcc_{i+1}_mean'] = np.mean(delta2_mfccs[i])
            features[f'delta2_mfcc_{i+1}_std'] = np.std(delta2_mfccs[i])
        
        return features
    
    def extract_wavelet_features(self, audio, wavelet='db4', level=5):
        """
        Dalgacık dönüşümü öznitelikleri çıkar
        
        Args:
            audio (numpy.ndarray): Ses verisi
            wavelet (str): Dalgacık türü
            level (int): Dalgacık dönüşümü seviyesi
            
        Returns:
            dict: Dalgacık özellikleri
        """
        features = {}
        
        try:
            import pywt
            
            # Dalgacık dönüşümü uygula
            coeffs = pywt.wavedec(audio, wavelet, level=level)
            
            # Her seviye için özellikler
            for i, coeff in enumerate(coeffs):
                if i == 0:
                    name = 'approximation'
                else:
                    name = f'detail_{i}'
                
                features[f'wavelet_{name}_mean'] = np.mean(coeff)
                features[f'wavelet_{name}_std'] = np.std(coeff)
                features[f'wavelet_{name}_energy'] = np.sum(coeff**2)
                features[f'wavelet_{name}_entropy'] = -np.sum(coeff**2 * np.log(coeff**2 + 1e-10))
        
        except ImportError:
            print("PyWavelets kütüphanesi bulunamadı. Dalgacık özellikleri çıkarılamadı.")
        
        return features
    
    def extract_all_features(self, audio):
        """
        Tüm öznitelikleri çıkar
        
        Args:
            audio (numpy.ndarray): Ses verisi
            
        Returns:
            dict: Tüm öznitelikler
        """
        all_features = {}
        
        # Tüm öznitelik setlerini birleştir
        all_features.update(self.extract_time_domain_features(audio))
        all_features.update(self.extract_frequency_domain_features(audio))
        all_features.update(self.extract_mfcc_features(audio))
        all_features.update(self.extract_wavelet_features(audio))
        
        return all_features
    
    def extract_features_from_segments(self, segments):
        """
        Segment listesinden öznitelik çıkar
        
        Args:
            segments (list): Segment listesi
            
        Returns:
            pandas.DataFrame: Öznitelik matrisi
        """
        all_features = []
        
        for i, segment in enumerate(tqdm(segments, desc="Öznitelik çıkarılıyor")):
            features = self.extract_all_features(segment)
            features['segment_id'] = i
            all_features.append(features)
        
        # Öznitelikleri DataFrame'e dönüştür
        features_df = pd.DataFrame(all_features)
        
        return features_df
    
    def scale_features(self, features_df, fit=True):
        """
        Öznitelikleri ölçeklendir
        
        Args:
            features_df (pandas.DataFrame): Öznitelik matrisi
            fit (bool): True ise scaler'ı yeni veriye göre ayarla, False ise mevcut scaler'ı kullan
            
        Returns:
            pandas.DataFrame: Ölçeklendirilmiş öznitelik matrisi
        """
        # Segment_id gibi öznitelik olmayan sütunları ayır
        non_feature_cols = ['segment_id']
        feature_cols = [col for col in features_df.columns if col not in non_feature_cols]
        
        # Öznitelikleri ölçeklendir
        if fit:
            scaled_features = self.scaler.fit_transform(features_df[feature_cols])
        else:
            scaled_features = self.scaler.transform(features_df[feature_cols])
        
        # DataFrame'e dönüştür
        scaled_df = pd.DataFrame(scaled_features, columns=feature_cols)
        
        # Öznitelik olmayan sütunları geri ekle
        for col in non_feature_cols:
            if col in features_df.columns:
                scaled_df[col] = features_df[col].values
        
        return scaled_df
    
    def plot_feature_distributions(self, features_df, n_features=10):
        """
        Öznitelik dağılımlarını görselleştir
        
        Args:
            features_df (pandas.DataFrame): Öznitelik matrisi
            n_features (int): Görselleştirilecek öznitelik sayısı
        """
        # Segment_id gibi öznitelik olmayan sütunları çıkar
        non_feature_cols = ['segment_id']
        feature_cols = [col for col in features_df.columns if col not in non_feature_cols]
        
        # Rastgele n_features kadar öznitelik seç
        if len(feature_cols) > n_features:
            selected_features = np.random.choice(feature_cols, n_features, replace=False)
        else:
            selected_features = feature_cols
        
        # Görselleştir
        fig, axs = plt.subplots(n_features, 1, figsize=(10, 2*n_features))
        
        for i, feature in enumerate(selected_features):
            axs[i].hist(features_df[feature], bins=30)
            axs[i].set_title(feature)
            axs[i].set_ylabel('Frekans')
        
        plt.tight_layout()
        plt.show()


# Test için örnek kullanım
if __name__ == "__main__":
    from data_loader import HeartSoundDataLoader
    from preprocessing import HeartSoundPreprocessor
    
    # Veri yükleyici oluştur
    loader = HeartSoundDataLoader("../data")
    
    try:
        # Kayıt listesini yükle
        records = loader.load_records_list()
        
        if records:
            # İlk kaydı yükle
            first_record = records[0]
            audio_data, fs = loader.load_audio(first_record)
            
            if audio_data is not None:
                print(f"Kayıt: {first_record}")
                print(f"Örnekleme hızı: {fs} Hz")
                print(f"Kayıt uzunluğu: {len(audio_data) / fs:.2f} saniye")
                
                # Önişleyici oluştur
                preprocessor = HeartSoundPreprocessor(target_sr=2000)
                
                # Ses verisini işle
                processed_audio = preprocessor.process_audio(audio_data, fs)
                
                # Ses verisini segmentlere ayır
                segments = preprocessor.segment(processed_audio, segment_length_sec=3.0, overlap_ratio=0.5)
                
                print(f"Segment sayısı: {len(segments)}")
                
                # Öznitelik çıkarıcı oluştur
                feature_extractor = HeartSoundFeatureExtractor(target_sr=preprocessor.target_sr)
                
                # İlk segmentten öznitelikleri çıkar
                features = feature_extractor.extract_all_features(segments[0])
                
                print(f"Çıkarılan öznitelik sayısı: {len(features)}")
                
                # Zamansal öznitelikleri ekrana yazdır
                time_features = feature_extractor.extract_time_domain_features(segments[0])
                print("\nZaman Domaini Özellikleri:")
                for key, value in time_features.items():
                    print(f"{key}: {value:.4f}")
                
                # Tüm segmentlerden öznitelik çıkar
                features_df = feature_extractor.extract_features_from_segments(segments[:5])  # İlk 5 segment için
                
                print(f"\nÖznitelik Matrisi Boyutu: {features_df.shape}")
                
                # Öznitelikleri ölçeklendir
                scaled_features_df = feature_extractor.scale_features(features_df)
                
                # Öznitelik dağılımlarını görselleştir
                feature_extractor.plot_feature_distributions(features_df, n_features=5)
                
                # Ölçeklendirilmiş öznitelik dağılımlarını görselleştir
                feature_extractor.plot_feature_distributions(scaled_features_df, n_features=5)
    
    except Exception as e:
        print(f"Hata oluştu: {str(e)}") 