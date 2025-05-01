import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from oemer.inference import inference
from oemer.classifier import predict

# ================================
# Ayarlar (kendine göre güncelle)
# ================================
MODEL_PATH = "/content/oemer/checkpoints/seg_net"  # Eğittiğin model klasörü
IMG_PATH = "/content/drive/MyDrive/oemer_dataset/trainedmodel/inverted_niho.png"  # Test görselin
CLASSIFIER_NAME = "note"  # classifier.py içinde eğitilmiş model adı (note.model)
TARGET_CLASS_ID = 1  # segmentasyonda 'notehead' label ID'si (gerekirse değiştir)

# ================================
# Segmentasyon ve Preprocessing
# ================================
print("[INFO] Segmentasyon yapılıyor...")
class_map, _ = inference(
    model_path=MODEL_PATH,
    img_path=IMG_PATH,
    step_size=128,
    batch_size=16,
    manual_th=None,
    use_tf=True,
)

# ================================
# BBox çıkarma
# ================================
print("[INFO] Bounding-box çıkarılıyor...")
binary_mask = (class_map == TARGET_CLASS_ID).astype(np.uint8)
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bboxes = [cv2.boundingRect(c) for c in contours]

# ================================
# Region sınıflandırma
# ================================
print("[INFO] Region sınıflandırması başlıyor...")
image_gray = Image.open(IMG_PATH).convert("L")
img_np = np.array(image_gray)

results = []
for (x, y, w, h) in bboxes:
    region = img_np[y:y+h, x:x+w]
    try:
        label = predict(region, CLASSIFIER_NAME)
        results.append(((x, y, w, h), label))
    except Exception as e:
        print(f"[WARN] Sınıflandıramadı: {e}")
        continue

# ================================
# Görselleştirme
# ================================
print(f"[INFO] Toplam {len(results)} notaya etiket verildi.")
img_draw = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

for (x, y, w, h), label in results:
    cv2.rectangle(img_draw, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img_draw, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)

plt.figure(figsize=(14, 10))
plt.imshow(img_draw)
plt.axis("off")
plt.title("Sınıflandırılmış Notalar (4'lük, 8'lik, 16'lık)")
plt.show()
