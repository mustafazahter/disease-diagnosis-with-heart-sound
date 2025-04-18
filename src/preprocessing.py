"""
Kalp sesi kayıtlarını işlemek için önişleme modülü
"""

import os
import numpy as np
import pandas as pd
from scipy import signal
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import butter, cheby1, cheby2, ellip, lfilter, resample

class HeartSoundPreprocessor:
    """Kalp sesi önişleme sınıfı"""
    
    def __init__(self, target_sr=2000, filter_type="butter", filter_order=4):
        """
        Args:
            target_sr (int): Hedef örnekleme hızı
            filter_type (str): Filtre tipi ('butter', 'cheby1', 'cheby2', 'ellip')
            filter_order (int): Filtre derecesi
        """
        self.target_sr = target_sr
        self.filter_type = filter_type
        self.filter_order = filter_order
    
    def clean_audio(self, audio_data):
        """
        Ses verisindeki NaN ve Infinity değerlerini temizle
        
        Args:
            audio_data (numpy.ndarray): Ses verisi
            
        Returns:
            numpy.ndarray: Temizlenmiş ses verisi
        """
        # NaN değerleri 0 ile değiştir
        if np.isnan(audio_data).any():
            print("UYARI: Ses verisinde NaN değerleri tespit edildi ve temizlendi.")
            audio_data = np.nan_to_num(audio_data, nan=0.0)
        
        # Infinity değerleri maksimum float değeri ile değiştir
        if np.isinf(audio_data).any():
            print("UYARI: Ses verisinde Infinity değerleri tespit edildi ve temizlendi.")
            audio_data = np.nan_to_num(audio_data, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
        
        return audio_data
    
    def filter_audio(self, audio_data, fs, lowcut=25, highcut=400):
        """
        Ses verisini bant geçiren filtre ile filtrele
        
        Args:
            audio_data (numpy.ndarray): Ses verisi
            fs (float): Örnekleme hızı
            lowcut (float): Alçak kesim frekansı
            highcut (float): Yüksek kesim frekansı
            
        Returns:
            numpy.ndarray: Filtrelenmiş ses verisi
        """
        # Temizle
        audio_data = self.clean_audio(audio_data)
        
        # Normalize, eğer verinin bir aralığı varsa
        if np.ptp(audio_data) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Nyquist frekansı
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        # Bant geçiren filtre tasarımı
        if self.filter_type == "butter":
            b, a = butter(self.filter_order, [low, high], btype='band')
        elif self.filter_type == "cheby1":
            b, a = cheby1(self.filter_order, 0.5, [low, high], btype='band')
        elif self.filter_type == "cheby2":
            b, a = cheby2(self.filter_order, 30, [low, high], btype='band')
        elif self.filter_type == "ellip":
            b, a = ellip(self.filter_order, 0.5, 30, [low, high], btype='band')
        else:
            raise ValueError(f"Bilinmeyen filtre tipi: {self.filter_type}")
        
        # Filtreleme
        try:
            y = lfilter(b, a, audio_data)
            return y
        except Exception as e:
            print(f"Filtreleme sırasında hata oluştu: {str(e)}")
            return audio_data  # Hata durumunda orijinal veriyi döndür
    
    def resample(self, audio_data, fs):
        """
        Ses verisini yeniden örnekle
        
        Args:
            audio_data (numpy.ndarray): Ses verisi
            fs (float): Örnekleme hızı
            
        Returns:
            numpy.ndarray: Yeniden örneklenmiş ses verisi
        """
        if fs == self.target_sr:
            return audio_data
        
        # Temizle
        audio_data = self.clean_audio(audio_data)
        
        # Yeniden örnekleme faktörü
        factor = self.target_sr / fs
        
        # Yeni uzunluk
        new_length = int(len(audio_data) * factor)
        
        # Yeniden örnekle
        try:
            resampled = resample(audio_data, new_length)
            return resampled
        except Exception as e:
            print(f"Yeniden örnekleme sırasında hata oluştu: {str(e)}")
            # Hata durumunda, basit bir lineer interpolasyon yaparak yeniden örnekle
            x_old = np.linspace(0, 1, len(audio_data))
            x_new = np.linspace(0, 1, new_length)
            resampled = np.interp(x_new, x_old, audio_data)
            return resampled
    
    def process_audio(self, audio_data, fs):
        """
        Ses verisini işle: filtrele, yeniden örnekle
        
        Args:
            audio_data (numpy.ndarray): Ses verisi
            fs (float): Örnekleme hızı
            
        Returns:
            numpy.ndarray: İşlenmiş ses verisi
        """
        # Temizlik kontrolü ve temizleme
        audio_data = self.clean_audio(audio_data)
        
        # Filtreleme
        filtered_data = self.filter_audio(audio_data, fs)
        
        # Yeniden örnekleme
        resampled_data = self.resample(filtered_data, fs)
        
        # Son temizlik
        clean_data = self.clean_audio(resampled_data)
        
        return clean_data
    
    def segment(self, audio_data, segment_length_sec, overlap_ratio=0.5):
        """
        Ses verisini segmentlere ayır
        
        Args:
            audio_data (numpy.ndarray): Ses verisi
            segment_length_sec (float): Segment uzunluğu (saniye)
            overlap_ratio (float): Örtüşme oranı (0-1 arası)
            
        Returns:
            list: Ses segmentleri listesi
        """
        # Segment uzunluğu (örnek sayısı)
        segment_length = int(segment_length_sec * self.target_sr)
        
        # Adım boyutu (örnek sayısı)
        step_size = int(segment_length * (1 - overlap_ratio))
        
        segments = []
        
        # Ses verisini segmentlere ayır
        for start in range(0, len(audio_data) - segment_length + 1, step_size):
            segment = audio_data[start:start + segment_length].copy()
            
            # Segmenti temizle
            segment = self.clean_audio(segment)
            
            # Normalize
            if np.ptp(segment) > 0:  # Eğer segment sabit değilse
                segment = segment / np.max(np.abs(segment))
            
            segments.append(segment)
        
        return segments
    
    def denoise(self, audio, n_grad_freq=2, n_grad_time=4, n_fft=2048,
                win_length=2048, hop_length=512, n_std_thresh=1.5):
        """
        Spectral gating ile gürültü azaltma
        
        Args:
            audio (numpy.ndarray): Ses verisi
            n_grad_freq (int): Spectral gating parametresi
            n_grad_time (int): Spectral gating parametresi
            n_fft (int): FFT boyutu
            win_length (int): Pencere uzunluğu
            hop_length (int): Hop uzunluğu
            n_std_thresh (float): Gürültü eşiği
            
        Returns:
            numpy.ndarray: Gürültüsü azaltılmış ses verisi
        """
        # STFT ile ses verisini frekans domaine dönüştür
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length,
                           win_length=win_length)
        stft_db = librosa.amplitude_to_db(np.abs(stft))
        
        # Gürültü profilini hesapla
        mean_freq_noise = np.mean(stft_db, axis=1)
        std_freq_noise = np.std(stft_db, axis=1)
        noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
        
        # Maske oluştur
        mask = np.ones(stft.shape, dtype=bool)
        for i in range(mask.shape[0]):
            mask[i, :] = stft_db[i, :] < noise_thresh[i]
        
        # Maskeyi uygula
        stft_denoised = stft.copy()
        stft_denoised[mask] = 0
        
        # ISTFT ile zaman domaine geri dönüştür
        audio_denoised = librosa.istft(stft_denoised, hop_length=hop_length,
                                     win_length=win_length)
        
        return audio_denoised
    
    def plot_preprocessing_steps(self, audio, original_sr):
        """
        Önişleme adımlarını görselleştir
        
        Args:
            audio (numpy.ndarray): Orijinal ses verisi
            original_sr (int): Orijinal örnekleme hızı
        """
        # Örnekleme hızını değiştir
        audio_resampled = self.resample(audio, original_sr)
        
        # Normalize et
        audio_normalized = self.clean_audio(audio_resampled)
        
        # Bandpass filtre uygula
        audio_filtered = self.filter_audio(audio_normalized, original_sr)
        
        # Gürültü azalt
        audio_denoised = self.denoise(audio_filtered)
        
        # Görselleştir
        fig, axs = plt.subplots(4, 1, figsize=(12, 10))
        
        # Zaman dizilerini oluştur
        t_original = np.linspace(0, len(audio) / original_sr, len(audio))
        t_processed = np.linspace(0, len(audio_denoised) / self.target_sr, len(audio_denoised))
        
        # Orijinal
        axs[0].plot(t_original, audio)
        axs[0].set_title('Orijinal Ses')
        axs[0].set_xlabel('Zaman (s)')
        axs[0].set_ylabel('Genlik')
        
        # Normalize edilmiş
        axs[1].plot(t_processed, audio_normalized)
        axs[1].set_title('Normalize Edilmiş Ses')
        axs[1].set_xlabel('Zaman (s)')
        axs[1].set_ylabel('Genlik')
        
        # Filtrelenmiş
        axs[2].plot(t_processed, audio_filtered)
        axs[2].set_title('Bandpass Filtrelenmiş Ses')
        axs[2].set_xlabel('Zaman (s)')
        axs[2].set_ylabel('Genlik')
        
        # Gürültüsü azaltılmış
        axs[3].plot(t_processed, audio_denoised)
        axs[3].set_title('Gürültüsü Azaltılmış Ses')
        axs[3].set_xlabel('Zaman (s)')
        axs[3].set_ylabel('Genlik')
        
        plt.tight_layout()
        plt.show()
        
        # Spektrogramları da görselleştir
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        
        # Orijinal spektrogram
        S_original = librosa.feature.melspectrogram(y=audio_resampled, sr=self.target_sr)
        S_db_original = librosa.power_to_db(S_original, ref=np.max)
        librosa.display.specshow(S_db_original, x_axis='time', y_axis='mel', sr=self.target_sr, ax=axs[0])
        axs[0].set_title('Orijinal Spektrogram')
        
        # İşlenmiş spektrogram
        S_processed = librosa.feature.melspectrogram(y=audio_denoised, sr=self.target_sr)
        S_db_processed = librosa.power_to_db(S_processed, ref=np.max)
        librosa.display.specshow(S_db_processed, x_axis='time', y_axis='mel', sr=self.target_sr, ax=axs[1])
        axs[1].set_title('İşlenmiş Spektrogram')
        
        plt.tight_layout()
        plt.show()


# Test için örnek kullanım
if __name__ == "__main__":
    from data_loader import HeartSoundDataLoader
    
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
                
                # Önişleme adımlarını görselleştir
                preprocessor.plot_preprocessing_steps(audio_data, fs)
                
                # Ses verisini segmentlere ayır
                processed_audio = preprocessor.process_audio(audio_data, fs)
                segments = preprocessor.segment(processed_audio, segment_length_sec=3.0, overlap_ratio=0.5)
                
                print(f"Segment sayısı: {len(segments)}")
                print(f"Segment uzunluğu: {len(segments[0]) / preprocessor.target_sr:.2f} saniye")
                
                # İlk segmenti görselleştir
                plt.figure(figsize=(10, 4))
                t = np.linspace(0, len(segments[0]) / preprocessor.target_sr, len(segments[0]))
                plt.plot(t, segments[0])
                plt.title(f"Segment 1/{len(segments)}")
                plt.xlabel("Zaman (s)")
                plt.ylabel("Genlik")
                plt.grid(True)
                plt.tight_layout()
                plt.show()
    
    except Exception as e:
        print(f"Hata oluştu: {str(e)}") 