import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Muat model YOLO
model = YOLO('best.pt')  # Pastikan path ini benar


# Fungsi untuk melakukan deteksi jerawat
def detect_acne(image):
    # Menggunakan BGR untuk OpenCV
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Deteksi jerawat
    results = model.predict(source=img_bgr, imgsz=640)

    # Menggambar kotak deteksi
    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes.xyxy) > 0:
            for i, box in enumerate(boxes.xyxy):
                x1, y1, x2, y2 = box[:4]
                conf = boxes.conf[i] if boxes.conf is not None and len(boxes.conf) > i else None

                # Menggambar kotak di gambar asli dengan warna merah
                cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 255), thickness=2)

                if conf is not None:
                    # Menambahkan label dengan confidence score
                    cv2.putText(img_bgr, f'{conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 2)

    # Mengembalikan gambar ke format RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


# Judul aplikasi
st.title('Deteksi Jerawat Menggunakan YOLOv8')

# Pilihan untuk mengunggah gambar
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

# Tampilkan gambar asli terlebih dahulu
if uploaded_file is not None:
    # Membaca gambar
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Tampilkan gambar asli
    st.image(image, caption='Gambar yang Diupload', use_column_width=True)

    # Tombol untuk melakukan deteksi jerawat
    if st.button('Deteksi Jerawat'):
        # Deteksi jerawat dalam gambar
        detected_image = detect_acne(image_np)

        # Tampilkan gambar hasil deteksi
        st.image(detected_image, caption='Hasil Deteksi Jerawat', use_column_width=True)