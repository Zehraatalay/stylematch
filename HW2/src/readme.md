# BIL468 HW2 Readme

Bu klasor, `Homework_2.pdf` icin hazirlanan `HOG + PyTorch MLP` tabanli pencere-siniflandirma odevini icerir.


- HW2 giris dosyasi: `main_hw2.py`
- HW1 ile karistirilmamasi gereken komutlar: `python main_hw2.py ...`
- Hazir veri ve hazir ciktilar repo icinde bulunur:
  - `data/hw2_prepared/`
  - `outputs/hw2/`

## Gerekli Ortam

- Python `3.11+` veya `3.12`
- Internet erisimi varsa veri seti tekrar indirilebilir
- Internet erisimi olmasa bile repo icindeki `data/hw2_prepared/` ile validation ve test calisabilir

Kurulum:

```bash
pip install -r requirements.txt
```

## HW2 Dosyalari

- `main_hw2.py`
  - `prepare-data`: Fashionpedia veri setini okuyup HW2 icin pencere-etiketli HOG verisi uretir
  - `run-validation`: MLP hiperparametre aramasi yapar
  - `run-test`: secilen model ile test metriklerini uretir
- `stylematch/hw2_data.py`
  - veri hazirlama, pencere etiketleme, HOG feature cikarma
- `stylematch/hw2_features.py`
  - `256x256` goruntu, `128x128` pencere, `64` stride ve `7560` boyutlu HOG tanimi
- `stylematch/hw2_model.py`
  - MLP modeli, egitim, early stopping, validation aramasi
- `stylematch/hw2_evaluation.py`
  - test metrikleri, confusion matrix, tahmin csv ve demo gorselleri

## En Hizli Kontrol

Eger repo icindeki hazir veriler kullanilacaksa sadece su iki komut yeterlidir:

```bash
python main_hw2.py run-validation --config-names cfg_03,cfg_05,cfg_06 --max-epochs 10
python main_hw2.py run-test
```

Bu komutlar su klasorleri gunceller:

- `outputs/hw2/validation/`
- `outputs/hw2/test/`

## Bastan Tam Calistirma

Veriyi yeniden hazirlamak dahil tum akisi bastan calistirmak icin:

```bash
python main_hw2.py prepare-data
python main_hw2.py run-validation --config-names cfg_03,cfg_05,cfg_06 --max-epochs 10
python main_hw2.py run-test
```

Tek komutta:

```bash
python main_hw2.py run-all-experiments
```

## Kucuk Veriyle Hizli Smoke Test

Kodun uctan uca calistigini hizlica gormek icin:

```bash
python main_hw2.py run-all-experiments --prepared-dir tiny_data --outputs-dir tiny_outputs --num-classes 2 --train-target-per-class 5 --validation-target-per-class 5 --test-target-per-class 5
```

Not:
- Bu komut sadece hizli kontrol icindir
- Nihai rapor sayilari icin kullanilmamalidir

## Beklenen Ana Ciktilar

Validation:

- `outputs/hw2/validation/best_config.json`
- `outputs/hw2/validation/grid_search.json`
- `outputs/hw2/validation/metrics.json`

Test:

- `outputs/hw2/test/metrics.json`
- `outputs/hw2/test/predictions.csv`
- `outputs/hw2/test/confusion_matrix.png`
- `outputs/hw2/test/demo_*.png`

`


## Onemli Not

- HW1 komutlari `main.py` uzerindedir
- HW2 komutlari `main_hw2.py` uzerindedir

