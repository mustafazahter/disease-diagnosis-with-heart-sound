# Kalp Sesi ile Murmur Tespiti - Kurulum Talimatları

Bu belge, kalp sesi tabanlı murmur tespit projesinin kurulumu için adım adım talimatları içermektedir.

## Kurulum Seçenekleri

Projeyi farklı yöntemlerle kurabilirsiniz. Tercih ettiğiniz yöntemi seçin:

### Seçenek 1: Otomatik Kurulum (Önerilen)

Proje içindeki otomatik kurulum scriptini kullanabilirsiniz:

```bash
# 1. Gerekli paketleri ve veri setini indir
python src/setup.py
```

### Seçenek 2: Manuel Kurulum

Adım adım manuel kurulum yapmak isterseniz:

```bash
# 1. Gerekli temel paketleri yükle
pip install --upgrade pip setuptools wheel

# 2. Projenin gereksinimlerini yükle
pip install -r requirements.txt

# 3. Veri setini indir
python src/setup.py --skip_requirements
```

### Seçenek 3: Sanal Ortam Oluşturarak Kurulum (Python Sürüm Sorunları Yaşıyorsanız)

Python 3.9, 3.10 veya 3.11 kullanarak bir sanal ortam oluşturabilirsiniz:

```bash
# 1. Sanal ortam oluştur (Python 3.10 örneği)
python -m venv venv --prompt="heart-sound"

# 2. Sanal ortamı etkinleştir (Windows)
venv\Scripts\activate

# 3. Gerekli paketleri yükle
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 4. Veri setini indir
python src/setup.py --skip_requirements
```

## Olası Sorunlar ve Çözümleri

### 1. Numpy Kurulumu Sorunu

Python 3.12 ile numpy 1.24.3 gibi eski sürümleri kullanırken sorun yaşanabilir:

```
Çözüm: requirements.txt dosyasını güncelleyin veya manuel olarak şunu çalıştırın:
pip install numpy>=1.26.0
```

### 2. TensorFlow Kurulumu Sorunu

TensorFlow'un Windows'ta çalışan en son sürümü Python 3.10'u destekler:

```
Çözüm: Python 3.10 ile sanal ortam oluşturun veya CPU-only sürümünü kurabilirsiniz:
pip install tensorflow-cpu
```

### 3. Veri Seti İndirme Sorunu

Veri seti büyük (yaklaşık 450 MB) olduğu için indirme sırasında sorun yaşanabilir:

```
Çözüm: Veri setini manuel olarak indirip data/ klasörüne yerleştirin:
https://physionet.org/content/circor-heart-sound/1.0.3/
```

## Kurulumu Doğrulama

Kurulumun doğru çalıştığını test etmek için:

```bash
# Test scriptini çalıştır
python src/data_loader.py
```

Bu script metadata ve ilk ses dosyasını yüklemeyi deneyecektir. Herhangi bir hata mesajı almazsanız kurulum başarılıdır. 