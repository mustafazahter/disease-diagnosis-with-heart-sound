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

class HeartSoundPreprocessor:
    """Kalp sesi önişleme sınıfı"""
    
    def __init__(self, target_sr=2000):
        """
        Args:
            target_sr (int): Hedef örnekleme hızı
        """
        self.target_sr = target_sr
    
    def resample(self, audio, original_sr):
        """
        Ses verisini hedef örnekleme hızına yeniden örnekle
        
        Args:
            audio (numpy.ndarray): Ses verisi
            original_sr (int): Orijinal örnekleme hızı
            
        Returns:
            numpy.ndarray: Yeniden örneklenmiş ses verisi
        """
        if original_sr != self.target_sr:
            return librosa.resample(audio, orig_sr=original_sr, target_sr=self.target_sr)
        return audio
    
    def normalize(self, audio):
        """
        Ses verisini normalize et
        
        Args:
            audio (numpy.ndarray): Ses verisi
            
        Returns:
            numpy.ndarray: Normalize edilmiş ses verisi
        """
        if np.std(audio) > 0:
            return (audio - np.mean(audio)) / np.std(audio)
        return audio
    
    def bandpass_filter(self, audio, lowcut=25, highcut=400):
        """
        Bandpass filtre uygula
        
        Args:
            audio (numpy.ndarray): Ses verisi
            lowcut (int): Alt kesim frekansı (Hz)
            highcut (int): Üst kesim frekansı (Hz)
            
        Returns:
            numpy.ndarray: Filtrelenmiş ses verisi
        """
        nyquist = 0.5 * self.target_sr
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Butterworth bandpass filtre tasarımı
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Filtreyi uygula
        return signal.filtfilt(b, a, audio)
    
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
    
    def segment(self, audio, segment_length_sec=3.0, overlap_ratio=0.5):
        """
        Ses verisini segmentlere ayır
        
        Args:
            audio (numpy.ndarray): Ses verisi
            segment_length_sec (float): Segment uzunluğu (saniye)
            overlap_ratio (float): Örtüşme oranı (0-1 arası)
            
        Returns:
            list: Segment listesi
        """
        # Segment uzunluğunu örnekleme sayısına dönüştür
        segment_length = int(segment_length_sec * self.target_sr)
        
        # Örtüşme miktarını hesapla
        hop_length = int(segment_length * (1 - overlap_ratio))
        
        # Segment sayısını hesapla
        n_segments = 1 + (len(audio) - segment_length) // hop_length
        
        segments = []
        for i in range(n_segments):
            start = i * hop_length
            end = start + segment_length
            
            # Eğer segment, ses verisinin sonunu aşıyorsa döngüyü sonlandır
            if end > len(audio):
                break
                
            segment = audio[start:end]
            segments.append(segment)
        
        return segments
    
    def process_audio(self, audio, original_sr):
        """
        Ses verisine tüm işlemleri uygula
        
        Args:
            audio (numpy.ndarray): Ses verisi
            original_sr (int): Orijinal örnekleme hızı
            
        Returns:
            numpy.ndarray: İşlenmiş ses verisi
        """
        # Örnekleme hızını değiştir
        audio_resampled = self.resample(audio, original_sr)
        
        # Normalize et
        audio_normalized = self.normalize(audio_resampled)
        
        # Bandpass filtre uygula
        audio_filtered = self.bandpass_filter(audio_normalized)
        
        # Gürültü azalt
        audio_denoised = self.denoise(audio_filtered)
        
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
        audio_normalized = self.normalize(audio_resampled)
        
        # Bandpass filtre uygula
        audio_filtered = self.bandpass_filter(audio_normalized)
        
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