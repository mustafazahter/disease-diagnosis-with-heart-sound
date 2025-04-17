"""
PhysioNet CirCor DigiScope Phonocardiogram veri setini yüklemek için modül.
"""

import os
import pandas as pd
import numpy as np
import wfdb
from tqdm import tqdm
import matplotlib.pyplot as plt


class HeartSoundDataLoader:
    """Kalp sesi veri seti için veri yükleme sınıfı"""
    
    def __init__(self, data_dir, metadata_file='training_data.csv'):
        """
        Args:
            data_dir (str): Veri klasörü yolu
            metadata_file (str): Metadata dosyası ismi
        """
        self.data_dir = data_dir
        self.metadata_path = os.path.join(data_dir, metadata_file)
        self.metadata = None
        self.records = None
        
    def load_metadata(self):
        """Metadata dosyasını yükle"""
        if os.path.exists(self.metadata_path):
            self.metadata = pd.read_csv(self.metadata_path)
            print(f"Metadata yüklendi: {len(self.metadata)} kayıt bulundu.")
            return self.metadata
        else:
            raise FileNotFoundError(f"Metadata dosyası bulunamadı: {self.metadata_path}")
    
    def load_records_list(self):
        """RECORDS dosyasından kayıt listesini yükle"""
        records_path = os.path.join(self.data_dir, 'RECORDS')
        if os.path.exists(records_path):
            with open(records_path, 'r') as f:
                self.records = [line.strip() for line in f]
            print(f"Kayıt listesi yüklendi: {len(self.records)} kayıt bulundu.")
            return self.records
        else:
            raise FileNotFoundError(f"RECORDS dosyası bulunamadı: {records_path}")
    
    def load_audio(self, record_name):
        """
        Belirli bir kaydın ses verisini yükle
        
        Args:
            record_name (str): Kayıt ismi (örn: "50001_AV")
            
        Returns:
            tuple: (ses verisi, örnekleme hızı)
        """
        record_path = os.path.join(self.data_dir, 'training_data', record_name)
        
        try:
            # WFDB kütüphanesi ile .hea dosyasından kayıt bilgilerini oku
            record = wfdb.rdrecord(record_path)
            
            # Ses verisini ve örnekleme hızını döndür
            return record.p_signal.flatten(), record.fs
        except Exception as e:
            print(f"Hata: {record_name} yüklenirken bir sorun oluştu - {str(e)}")
            return None, None
    
    def plot_audio(self, audio_data, fs, title="Kalp Sesi Kaydı"):
        """
        Ses verisini görselleştir
        
        Args:
            audio_data (numpy.ndarray): Ses verisi
            fs (int): Örnekleme hızı
            title (str): Grafik başlığı
        """
        if audio_data is None:
            print("Görselleştirilecek ses verisi yok.")
            return
        
        duration = len(audio_data) / fs
        time = np.linspace(0, duration, len(audio_data))
        
        plt.figure(figsize=(12, 4))
        plt.plot(time, audio_data)
        plt.title(title)
        plt.xlabel("Zaman (s)")
        plt.ylabel("Genlik")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def load_batch(self, record_names, max_length=None):
        """
        Bir grup kayıt için ses verilerini yükle
        
        Args:
            record_names (list): Kayıt isimleri listesi
            max_length (int, optional): Tüm kayıtların standardize edileceği maksimum uzunluk
            
        Returns:
            tuple: (ses verileri dizisi, örnekleme hızları listesi)
        """
        audio_data_list = []
        fs_list = []
        
        for record_name in tqdm(record_names, desc="Veri yükleniyor"):
            audio_data, fs = self.load_audio(record_name)
            
            if audio_data is not None:
                # Eğer max_length belirtilmişse, tüm kayıtları aynı uzunluğa getir
                if max_length is not None:
                    if len(audio_data) > max_length:
                        audio_data = audio_data[:max_length]
                    else:
                        # Eksik kısmı sıfırla doldur (padding)
                        padding = np.zeros(max_length - len(audio_data))
                        audio_data = np.concatenate([audio_data, padding])
                
                audio_data_list.append(audio_data)
                fs_list.append(fs)
        
        return np.array(audio_data_list), fs_list


# Test için örnek kullanım
if __name__ == "__main__":
    loader = HeartSoundDataLoader("../data")
    
    try:
        metadata = loader.load_metadata()
        print("Metadata sütunları:", metadata.columns.tolist())
        
        records = loader.load_records_list()
        print("İlk 5 kayıt:", records[:5])
        
        # İlk kaydı yükle ve görselleştir
        if records:
            first_record = records[0]
            audio_data, fs = loader.load_audio(first_record)
            
            if audio_data is not None:
                print(f"Kayıt: {first_record}")
                print(f"Örnekleme hızı: {fs} Hz")
                print(f"Kayıt uzunluğu: {len(audio_data) / fs:.2f} saniye")
                
                loader.plot_audio(audio_data, fs, title=f"Kalp Sesi Kaydı - {first_record}")
    
    except Exception as e:
        print(f"Hata oluştu: {str(e)}") 