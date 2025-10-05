import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

print("AI modeli eğitimi başlıyor...")

# Veri setini yükle
try:
    df = pd.read_csv('nasa_impact_dataset.csv')
except FileNotFoundError:
    print("HATA: 'nasa_impact_dataset.csv' dosyası bulunamadı.")
    print("Lütfen önce 'create_dataset_from_nasa.py' script'ini çalıştırın.")
    exit()

# Girdiler (X) ve Hedef (y) olarak ayır
features = ['mass_kg', 'velocity_kms', 'angle_deg', 'density']
target = 'crater_diameter_m'

X = df[features]
y = df[target]

# Veriyi eğitim ve test setleri olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Eğitim için {len(X_train)}, test için {len(X_test)} örnek veri ayrıldı.")

# Modeli oluştur ve eğit
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

print("Model eğitiliyor...")
gbr_model.fit(X_train, y_train)

# Modelin performansını test et
print("Modelin test verileri üzerindeki performansı değerlendiriliyor...")
y_pred = gbr_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n--- Model Performans Raporu ---")
print(f"Ortalama Mutlak Hata (MAE): {mae:.2f} metre")
print(f"R-kare Skoru (R²): {r2:.4f}")
print("---------------------------------")
print(f"Açıklama: Modelimiz, test verilerindeki krater çaplarını ortalama olarak {mae:.2f} metre hata ile tahmin etmektedir.")
print(f"R-kare skorunun {r2:.2f} olması, modelin veri setindeki değişkenliğin ~%{r2*100:.0f}'ini başarıyla açıkladığını gösterir.")

# Eğitilmiş modeli dosyaya kaydet
joblib.dump(gbr_model, 'impact_model.pkl')
print("\nEğitilmiş model 'impact_model.pkl' olarak kaydedildi. Artık Flask'ta kullanılabilir.")