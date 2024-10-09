import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

# Muat model YOLO
model = YOLO('best.pt')  # Pastikan path ini benar

# Fungsi untuk melakukan deteksi jerawat
def detect_acne(image):
    img_np = np.array(image)
    
    # Deteksi jerawat menggunakan YOLO
    results = model.predict(source=img_np, imgsz=640)

    # Gambar hasil deteksi dengan menggunakan Pillow
    draw = ImageDraw.Draw(image)

    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes.xyxy) > 0:
            for i, box in enumerate(boxes.xyxy):
                x1, y1, x2, y2 = box[:4]
                conf = boxes.conf[i] if boxes.conf is not None and len(boxes.conf) > i else None

                # Menggambar kotak deteksi di gambar asli
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

                if conf is not None:
                    # Menambahkan label dengan confidence score
                    draw.text((x1, y1), f'{conf:.2f}', fill="red")

    return image

# Judul aplikasi
st.title('Deteksi Jerawat Menggunakan YOLOv8')

# Pilihan untuk mengunggah gambar
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

# Tampilkan gambar asli terlebih dahulu
if uploaded_file is not None:
    # Membaca gambar
    image = Image.open(uploaded_file)

    # Tampilkan gambar asli
    st.image(image, caption='Gambar yang Diupload', use_column_width=True)

    # Tombol untuk melakukan deteksi jerawat
    if st.button('Deteksi Jerawat'):
        # Deteksi jerawat dalam gambar
        detected_image = detect_acne(image)

        # Tampilkan gambar hasil deteksi
        st.image(detected_image, caption='Hasil Deteksi Jerawat', use_column_width=True)
