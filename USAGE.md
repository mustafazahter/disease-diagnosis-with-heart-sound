# Kalp Sesi ile Murmur Tespiti Kullanım Kılavuzu

Bu doküman, kalp sesi tabanlı murmur tespit sisteminin kullanımını adım adım açıklamaktadır.

## 1. Kurulum

### 1.1. Gerekli Paketleri Yükleme

```bash
pip install -r requirements.txt
```

### 1.2. Veri Setini İndirme

Aşağıdaki komut, PhysioNet CirCor DigiScope veri setini otomatik olarak indirecek ve gerekli düzenlemeleri yapacaktır:

```bash
python src/setup.py
```

Alternatif olarak, veri setini manuel olarak şu adresden indirebilirsiniz: [PhysioNet CirCor DigiScope Dataset](https://physionet.org/content/circor-heart-sound/1.0.3/)

## 2. Veri Keşfi ve Analizi

Veri setini keşfetmek için hazırladığımız notebook'u çalıştırabilirsiniz:

```bash
jupyter notebook notebooks/EDA.ipynb
```

## 3. Model Eğitimi

Modeli eğitmek için aşağıdaki komutu kullanabilirsiniz:

```bash
python src/train.py --data_dir=data --model_type=cnn --feature_type=raw_waveform --epochs=50
```

### 3.1. Eğitim Parametreleri

- `--data_dir`: Veri setinin bulunduğu klasör
- `--model_type`: Kullanılacak model tipi (mlp, cnn, lstm, hybrid)
- `--feature_type`: Kullanılacak öznitelik tipi (raw_waveform, features, spectrogram)
- `--segment_length`: Segment uzunluğu (saniye)
- `--overlap_ratio`: Segment örtüşme oranı
- `--batch_size`: Batch boyutu
- `--epochs`: Epoch sayısı
- `--model_path`: Modelin kaydedileceği dosya yolu

## 4. Tahmin Yapma

Eğitilmiş modeli kullanarak bir kayıt için tahmin yapmak için:

```bash
python src/predict.py --record_name=50001_AV --model_path=models/heart_sound_model.h5 --visualize
```

veya bir ses dosyası için:

```bash
python src/predict.py --audio_path=path/to/audio.wav --model_path=models/heart_sound_model.h5 --visualize
```

### 4.1. Tahmin Parametreleri

- `--record_name`: PhysioNet veri setinden bir kayıt adı
- `--audio_path`: Bir ses dosyası yolu
- `--data_dir`: Veri klasörü yolu (record_name kullanılırsa)
- `--model_path`: Model dosyası yolu
- `--feature_extractor_path`: Öznitelik çıkarıcı model yolu (feature_type=features ise)
- `--feature_type`: Öznitelik tipi (raw_waveform, features, spectrogram)
- `--segment_length`: Segment uzunluğu (saniye)
- `--overlap_ratio`: Segment örtüşme oranı
- `--visualize`: Tahminleri görselleştir

## 5. Modüllerin Ayrı Ayrı Kullanımı

Projedeki her bir modülü ayrı ayrı kullanabilirsiniz:

### 5.1. Veri Yükleme

```python
from src.data_loader import HeartSoundDataLoader

loader = HeartSoundDataLoader("data")
metadata = loader.load_metadata()
records = loader.load_records_list()
audio_data, fs = loader.load_audio("50001_AV")
```

### 5.2. Önişleme

```python
from src.preprocessing import HeartSoundPreprocessor

preprocessor = HeartSoundPreprocessor(target_sr=2000)
processed_audio = preprocessor.process_audio(audio_data, fs)
segments = preprocessor.segment(processed_audio, segment_length_sec=3.0, overlap_ratio=0.5)
```

### 5.3. Öznitelik Çıkarma

```python
from src.feature_extraction import HeartSoundFeatureExtractor

feature_extractor = HeartSoundFeatureExtractor(target_sr=2000)
features = feature_extractor.extract_all_features(segments[0])
```

### 5.4. Model

```python
from src.model import HeartSoundClassifier

classifier = HeartSoundClassifier(model_type='cnn', input_shape=(6000, 1), num_classes=2)
model = classifier.build_model()
history = classifier.train(X_train, y_train, X_val, y_val, batch_size=32, epochs=50)
classifier.evaluate(X_test, y_test)
```

## 6. Sonuçları Değerlendirme

Eğitim sonuçlarını görselleştirmek için:

```python
classifier.plot_training_history(history)
classifier.plot_confusion_matrix(X_test, y_test, class_names=['Normal', 'Murmur'])
classifier.plot_roc_curve(X_test, y_test)
```