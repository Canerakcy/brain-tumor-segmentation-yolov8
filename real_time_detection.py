# real_time_detection.py
# YOLOv8 ile gerçek zamanlı webcam üzerinde beyin tümörü tespiti

from ultralytics import YOLO
import cv2

# 1. Eğitilmiş modeli yükle (modelin .pt dosya yolunu doğru gir)
model = YOLO("runs/detect/train/weights/best.pt")

# 2. Webcam başlat
cap = cv2.VideoCapture(0)  # 0 = varsayılan kamera

if not cap.isOpened():
    print("Kamera başlatılamadı!")
    exit()

print("Model yüklendi. Kamera başlatıldı. Gerçek zamanlı tespit başlıyor...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kare alınamadı!")
        break

    # 3. YOLOv8 ile tespit yap
    results = model.predict(source=frame, conf=0.4, save=False)

    # 4. Tahmin sonuçlarını görüntü üzerine çiz
    annotated_frame = results[0].plot()

    # 5. Ekranda göster
    cv2.imshow("Beyin Tümörü Tespiti - YOLOv8", annotated_frame)

    # 6. 'q' tuşuna basınca çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 7. Kamera ve pencereyi serbest bırak
cap.release()
cv2.destroyAllWindows()
