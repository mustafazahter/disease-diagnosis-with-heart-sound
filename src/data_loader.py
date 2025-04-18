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
    
    def __init__(self, data_dir, metadata_file=None):
        """
        Args:
            data_dir (str): Veri klasörü yolu
            metadata_file (str, optional): Metadata dosyası ismi veya tam yolu. 
                                          None ise otomatik arama yapılır.
        """
        self.data_dir = data_dir
        
        # Metadata dosyası belirtilmişse, tam yol oluştur
        if metadata_file is not None:
            # Eğer tam yol verilmişse (içinde / veya \ varsa) olduğu gibi kullan
            if '/' in metadata_file or '\\' in metadata_file:
                self.metadata_path = metadata_file
            else:
                self.metadata_path = os.path.join(data_dir, metadata_file)
        else:
            # Varsayılan olarak data_dir içinde training_data.csv dosyasını kabul et
            self.metadata_path = os.path.join(data_dir, 'training_data.csv')
        
        self.metadata = None
        self.records = None
        
    def load_metadata(self, ask_for_path=True):
        """
        Metadata dosyasını yükle
        
        Args:
            ask_for_path (bool): Dosya bulunamazsa kullanıcıdan yeni konum sor
            
        Returns:
            pandas.DataFrame: Metadata içeren DataFrame
        """
        # Birkaç olası metadata dosya konumunu dene
        possible_metadata_paths = [
            self.metadata_path,
            os.path.join(self.data_dir, 'training_data.csv'),
            os.path.join(self.data_dir, 'files', 'training_data.csv'),
            os.path.join(self.data_dir, 'metadata', 'training_data.csv')
        ]
        
        for path in possible_metadata_paths:
            if os.path.exists(path):
                try:
                    self.metadata = pd.read_csv(path)
                    self.metadata_path = path
                    print(f"Metadata yüklendi: {len(self.metadata)} kayıt bulundu (konum: {path}).")
                    
                    # Metadata'nın ilk birkaç satırını göster
                    print("Metadata örneği:")
                    print(self.metadata.head(3))
                    return self.metadata
                except Exception as e:
                    print(f"Metadata dosyası okunurken hata oluştu: {str(e)}")
        
        # Hiçbir yoldan yüklenemedi, olası yolları göster
        print("\nMetadata dosyası bulunamadı. Aranan konumlar:")
        for path in possible_metadata_paths:
            print(f"  - {path}: {'MEVCUT' if os.path.exists(path) else 'YOK'}")
        
        if not ask_for_path:
            raise FileNotFoundError(f"Metadata dosyası bulunamadı: {self.metadata_path}")
        
        # Dosya bulunamadıysa ve ask_for_path=True ise kullanıcıdan yeni bir yol iste
        if ask_for_path:
            print("\nMetadata dosyası yolunu manuel olarak belirtebilirsiniz.")
            print("Örnek: C:/Users/kullanici/veri/training_data.csv")
            print("İptal etmek için 'iptal' yazın.\n")
            
            while True:
                user_path = input("Metadata dosyası yolunu girin: ")
                
                if user_path.lower() == 'iptal':
                    print("İşlem iptal edildi.")
                    return None
                
                if os.path.exists(user_path):
                    try:
                        self.metadata = pd.read_csv(user_path)
                        self.metadata_path = user_path
                        print(f"Metadata yüklendi: {len(self.metadata)} kayıt bulundu.")
                        
                        # Metadata'nın ilk birkaç satırını göster
                        print("Metadata örneği:")
                        print(self.metadata.head(3))
                        
                        # Data_dir'i güncelle
                        self.data_dir = os.path.dirname(user_path)
                        print(f"Veri dizini güncellendi: {self.data_dir}")
                        
                        return self.metadata
                    except Exception as e:
                        print(f"Hata: Dosya yüklenemedi - {str(e)}")
                else:
                    print(f"Hata: Dosya bulunamadı - {user_path}")
                    print("Lütfen geçerli bir dosya yolu girin.")
        
        return None
    
    def load_records_list(self, ask_for_path=True):
        """
        RECORDS dosyasından kayıt listesini yükle
        
        Args:
            ask_for_path (bool): Dosya bulunamazsa kullanıcıdan yeni konum sor
            
        Returns:
            list: Kayıt isimleri listesi
        """
        # Birkaç olası RECORDS dosya konumunu dene
        possible_records_paths = [
            os.path.join(self.data_dir, 'RECORDS'),
            os.path.join(self.data_dir, 'files', 'RECORDS'),
            os.path.join(self.data_dir, 'training_data', 'RECORDS')
        ]
        
        for path in possible_records_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        self.records = [line.strip() for line in f]
                    print(f"Kayıt listesi yüklendi: {len(self.records)} kayıt bulundu (konum: {path}).")
                    if len(self.records) > 0:
                        print(f"Örnek kayıt isimleri: {self.records[:5]}")
                    return self.records
                except Exception as e:
                    print(f"RECORDS dosyası okunurken hata oluştu: {str(e)}")
        
        # Hiçbir yoldan yüklenemedi, olası yolları göster
        print("\nRECORDS dosyası bulunamadı. Aranan konumlar:")
        for path in possible_records_paths:
            print(f"  - {path}: {'MEVCUT' if os.path.exists(path) else 'YOK'}")
        
        # Alternatif olarak, ses dosyalarını otomatik tarama işlemini dene
        audio_extensions = ['.hea', '.dat']
        print("\nRECORDS dosyası bulunamadı. Ses dosyalarını otomatik taramaya çalışılıyor...")
        
        # Olası ses dosyaları dizinlerini tara
        search_dirs = [
            os.path.join(self.data_dir, 'training_data'),
            self.data_dir,
            os.path.join(self.data_dir, 'files')
        ]
        
        records = []
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                print(f"Dizin taranıyor: {search_dir}")
                try:
                    # .hea uzantılı dosyaları bul
                    hea_files = [f for f in os.listdir(search_dir) if f.endswith('.hea')]
                    if hea_files:
                        # Dosya isimlerini uzantı olmadan kaydet
                        records = [os.path.splitext(f)[0] for f in hea_files]
                        self.records = records
                        print(f"Otomatik tarama ile {len(records)} kayıt bulundu.")
                        if len(records) > 0:
                            print(f"Örnek kayıt isimleri: {records[:5]}")
                        return records
                except Exception as e:
                    print(f"Dizin taraması başarısız: {str(e)}")
        
        if not ask_for_path:
            raise FileNotFoundError(f"RECORDS dosyası bulunamadı ve otomatik tarama başarısız oldu")
        
        # Dosya bulunamadıysa ve ask_for_path=True ise kullanıcıdan yeni bir yol iste
        if ask_for_path:
            print("\nRECORDS dosyası yolunu manuel olarak belirtebilirsiniz.")
            print("Örnek: C:/Users/kullanici/veri/RECORDS")
            print("İptal etmek için 'iptal' yazın.\n")
            
            while True:
                user_path = input("RECORDS dosyası yolunu girin: ")
                
                if user_path.lower() == 'iptal':
                    print("İşlem iptal edildi.")
                    return None
                
                if os.path.exists(user_path):
                    try:
                        with open(user_path, 'r') as f:
                            self.records = [line.strip() for line in f]
                        
                        # Data_dir'i güncelle
                        self.data_dir = os.path.dirname(user_path)
                        print(f"Veri dizini güncellendi: {self.data_dir}")
                        
                        print(f"Kayıt listesi yüklendi: {len(self.records)} kayıt bulundu.")
                        if len(self.records) > 0:
                            print(f"Örnek kayıt isimleri: {self.records[:5]}")
                        return self.records
                    except Exception as e:
                        print(f"Hata: Dosya yüklenemedi - {str(e)}")
                else:
                    print(f"Hata: Dosya bulunamadı - {user_path}")
                    print("Lütfen geçerli bir dosya yolu girin.")
        
        return None
    
    def load_audio(self, record_name):
        """
        Belirli bir kaydın ses verisini yükle
        
        Args:
            record_name (str): Kayıt ismi (örn: "50001_AV" veya "training_data/50001_AV")
            
        Returns:
            tuple: (ses verisi, örnekleme hızı)
        """
        # Kayıt ID'si ve konumunu çıkar
        if '/' in record_name:
            # 'training_data/2530_AV' formatındaysa
            basename = record_name.split('/')[-1]  # '2530_AV'
            directory = '/'.join(record_name.split('/')[:-1])  # 'training_data'
        else:
            # '2530_AV' formatındaysa
            basename = record_name
            directory = ''
        
        # Olası yolları oluştur
        possible_paths = []
        
        # 1. Doğrudan verilen isimle tam yol
        if directory:
            possible_paths.append(os.path.join(self.data_dir, record_name))
        
        # 2. Sadece dosya adını kullanarak data_dir altında
        possible_paths.append(os.path.join(self.data_dir, basename))
        
        # 3. data_dir/training_data/ altında
        possible_paths.append(os.path.join(self.data_dir, 'training_data', basename))
        
        # 4. data_dir/files/ altında
        possible_paths.append(os.path.join(self.data_dir, 'files', basename))
        
        # 5. Doğrudan data_dir altında, dizin olmadan
        if directory:
            possible_paths.append(os.path.join(self.data_dir, basename))
        
        for record_path in possible_paths:
            try:
                # WFDB kütüphanesi ile .hea dosyasından kayıt bilgilerini oku
                record = wfdb.rdrecord(record_path)
                
                # Başarılı yolu yazdır
                print(f"Ses dosyası başarıyla yüklendi: {record_path}")
                
                # Ses verisini ve örnekleme hızını döndür
                return record.p_signal.flatten(), record.fs
            except Exception as e:
                if os.path.exists(record_path + '.hea'):
                    print(f"HATA: {record_path}.hea dosyası mevcut, ancak yüklenirken sorun oluştu: {str(e)}")
                # Bir sonraki yolu dene
        
        # Debug için daha fazla bilgi göster
        print(f"\nARAMA YAPILAN YOLLAR ({record_name} için):")
        for path in possible_paths:
            hea_exists = os.path.exists(path + '.hea')
            dat_exists = os.path.exists(path + '.dat')
            print(f"  - {path}.hea: {'MEVCUT' if hea_exists else 'YOK'}")
            print(f"  - {path}.dat: {'MEVCUT' if dat_exists else 'YOK'}")
        
        # Kullanıcıdan manuel konum isteyelim
        print(f"\nKayıt bulunamadı: {record_name}")
        print("Kayıt dosyasının (.hea) yolunu manuel olarak belirtebilirsiniz.")
        print("Örnek: C:/Users/kullanici/veri/training_data/50001_AV.hea")
        print("İptal etmek için 'iptal' yazın.\n")
        
        while True:
            user_path = input(f"{record_name} kaydının .hea uzantılı dosya yolunu girin: ")
            
            if user_path.lower() == 'iptal':
                print("İşlem iptal edildi.")
                return None, None
            
            if not user_path.endswith('.hea'):
                user_path = user_path + '.hea'
            
            base_path = user_path[:-4]  # .hea uzantısını kaldır
            
            try:
                record = wfdb.rdrecord(base_path)
                print(f"Kayıt başarıyla yüklendi: {record_name}")
                return record.p_signal.flatten(), record.fs
            except Exception as e:
                print(f"Hata: Kayıt yüklenemedi - {str(e)}")
                print("Lütfen geçerli bir dosya yolu girin.")
        
        print(f"Hata: {record_name} yüklenirken bir sorun oluştu")
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
    # Komut satırı argümanı alarak veri yolu belirlenebilir
    import argparse
    parser = argparse.ArgumentParser(description='Kalp sesi verilerini yükle')
    parser.add_argument('--data_dir', type=str, 
                      default='/content/disease-diagnosis-with-heart-sound/data/the-circor-digiscope-phonocardiogram-dataset-1.0.3/', 
                      help='Veri klasörü yolu')
    args = parser.parse_args()
    
    loader = HeartSoundDataLoader(args.data_dir)
    
    try:
        print(f"Veri dizini: {loader.data_dir}")
        
        # Kullanıcıdan giriş alarak metadata yükle
        metadata = loader.load_metadata(ask_for_path=True)
        if metadata is not None:
            print("Metadata sütunları:", metadata.columns.tolist())
        
        # Kullanıcıdan giriş alarak kayıt listesini yükle
        records = loader.load_records_list(ask_for_path=True)
        if records is not None:
            print("İlk 5 kayıt:", records[:5])
        
            # İlk kaydı yükle ve görselleştir
            first_record = records[0]
            audio_data, fs = loader.load_audio(first_record)
            
            if audio_data is not None:
                print(f"Kayıt: {first_record}")
                print(f"Örnekleme hızı: {fs} Hz")
                print(f"Kayıt uzunluğu: {len(audio_data) / fs:.2f} saniye")
                
                loader.plot_audio(audio_data, fs, title=f"Kalp Sesi Kaydı - {first_record}")
    
    except Exception as e:
        print(f"Hata oluştu: {str(e)}") 