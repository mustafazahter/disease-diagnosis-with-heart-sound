"""
PhysioNet CirCor DigiScope Phonocardiogram veri setini indirme ve kurulum scripti
"""

import os
import sys
import subprocess
import argparse
import zipfile
import shutil
import requests
from tqdm import tqdm


def download_file(url, destination):
    """
    Dosyayı indir ve ilerlemeyi göster
    
    Args:
        url (str): İndirilecek dosya URL'si
        destination (str): Kaydedilecek dosya yolu
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # HTTP hatalarını kontrol et
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        progress_bar = tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True,
            desc=f"İndiriliyor: {os.path.basename(destination)}"
        )
        
        with open(destination, 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        
        progress_bar.close()
        return True
    except Exception as e:
        print(f"İndirme hatası: {str(e)}")
        return False


def install_requirements():
    """Gerekli Python paketlerini yükle"""
    requirements_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'requirements.txt')
    
    if os.path.exists(requirements_path):
        print("Gerekli paketler yükleniyor...")
        try:
            # Önce setuptools ve wheel yükle
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'])
            # Sonra gereksinimleri yükle
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])
            print("Paket kurulumu tamamlandı!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Paket kurulumu hatası: {str(e)}")
            print("Paketleri manuel olarak yüklemeyi deneyin: pip install -r requirements.txt")
            return False
    else:
        print(f"Hata: {requirements_path} bulunamadı.")
        return False


def download_physionet_dataset(data_dir):
    """
    PhysioNet CirCor DigiScope Phonocardiogram veri setini indir
    
    Args:
        data_dir (str): Veri indirilecek klasör yolu
    """
    # PhysioNet veri seti URL'si (zip dosyası)
    url = "https://physionet.org/static/published-projects/circor-heart-sound/the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip"
    
    # Veri klasörünü oluştur
    os.makedirs(data_dir, exist_ok=True)
    
    # Zip dosyasının indirileceği yol
    zip_path = os.path.join(data_dir, "circor-heart-sound.zip")
    
    # Eğer zip dosyası daha önce indirilmediyse indir
    if not os.path.exists(zip_path):
        print(f"Veri seti indiriliyor: {url}")
        if not download_file(url, zip_path):
            print("Veri seti indirilemedi. Manuel olarak indirip 'data' klasörüne yerleştirebilirsiniz.")
            print("İndirme URL'si: https://physionet.org/content/circor-heart-sound/1.0.3/")
            return False
    else:
        print(f"Zip dosyası zaten mevcut: {zip_path}")
    
    # Zip dosyasını çıkar
    try:
        print("Zip dosyası açılıyor...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Dosyaları düzenle
        extracted_dir = os.path.join(data_dir, "circor-heart-sound-1.0.3")
        
        if os.path.exists(extracted_dir):
            # Dosyaları bir üst klasöre taşı
            for item in os.listdir(extracted_dir):
                src_path = os.path.join(extracted_dir, item)
                dst_path = os.path.join(data_dir, item)
                
                if os.path.exists(dst_path):
                    if os.path.isdir(dst_path):
                        shutil.rmtree(dst_path)
                    else:
                        os.remove(dst_path)
                
                shutil.move(src_path, data_dir)
            
            # Boş klasörü sil
            shutil.rmtree(extracted_dir)
        
        print(f"Veri seti hazır: {data_dir}")
        return True
    except Exception as e:
        print(f"Veri seti açma hatası: {str(e)}")
        return False


def setup_environment():
    """Projenin çalışacağı ortamı hazırla"""
    # Gerekli klasörleri oluştur
    dirs = ['data', 'models', 'notebooks']
    for dir_name in dirs:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), dir_name)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Klasör oluşturuldu: {dir_name}/")


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description='PhysioNet veri setini indir ve gerekli paketleri yükle')
    
    parser.add_argument('--data_dir', type=str, default='../data',
                      help='Veri indirilecek klasör yolu')
    parser.add_argument('--skip_requirements', action='store_true',
                      help='Paket kurulumunu atla')
    parser.add_argument('--skip_download', action='store_true',
                      help='Veri indirmeyi atla')
    
    args = parser.parse_args()
    
    # Ortamı hazırla
    setup_environment()
    
    success = True
    
    # Gerekli paketleri yükle
    if not args.skip_requirements:
        success = install_requirements() and success
    
    # Veri setini indir
    if not args.skip_download:
        success = download_physionet_dataset(args.data_dir) and success
    
    if success:
        print("Kurulum başarıyla tamamlandı!")
    else:
        print("Kurulum sırasında bazı sorunlar oluştu. Lütfen hata mesajlarını kontrol edin.")


if __name__ == "__main__":
    main() 