# Kalp Sesi ile Murmur Tespiti Projesi

Bu proje, kalp seslerinden kalp üfürümü (murmur) tespitini amaçlayan bir makine öğrenmesi modelini geliştirmeyi hedeflemektedir. [PhysioNet CirCor DigiScope Phonocardiogram veri seti](https://physionet.org/content/circor-heart-sound/1.0.3/) kullanılmaktadır.

## Proje Hakkında

Kalp sesleri içerisinde bulunan anormal sesler (murmur), kalp hastalıklarının erken teşhisinde büyük önem taşır. Bu projede, fonokardiogram kayıtları üzerinde derin öğrenme teknikleri kullanılarak murmur tespiti yapılmaktadır.

## Proje Yapısı

```
disease-diagnosis-with-heart-sound/
│
├── data/                     # Ham ve işlenmiş veri dosyaları
│
├── src/                      # Kaynak kodlar
│   ├── data_loader.py        # Veri yükleme modülü
│   ├── preprocessing.py      # Ön işleme fonksiyonları
│   ├── feature_extraction.py # Öznitelik çıkarma
│   ├── model.py              # Model mimarisi
│   └── train.py              # Eğitim kodu
│
├── models/                   # Eğitilmiş modeller
│
├── notebooks/                # Jupyter notebooks
│   ├── EDA.ipynb             # Keşifsel veri analizi
│   └── model_evaluation.ipynb # Model değerlendirme
│
├── requirements.txt          # Gerekli Python paketleri
└── README.md                 # Proje açıklaması
```

## Kurulum ve Kullanım

1. Gereksinimleri yükleyin:
```
pip install -r requirements.txt
```

2. PhysioNet'ten veri setini indirin ve `data/` klasörüne yerleştirin.

3. Veriyi önişleme ve özellikleri çıkarma:
```
python src/preprocessing.py
```

4. Modeli eğitme:
```
python src/train.py
```

5. Eğitilmiş modeli kullanarak tahmin yapma:
```
python src/predict.py
``` 