from django.shortcuts import render
import tensorflow as tf
import numpy as np
from django.core.files.storage import default_storage
from django.http import JsonResponse
from tensorflow.keras.preprocessing import image
from .models import ulos
from django.shortcuts import get_object_or_404

# Load model saat Django dimulai
model_path = "klasifikasi_ulos/model/model_ulos.keras"
model = tf.keras.models.load_model(model_path)

def predict_image(request):
    if request.method == "POST":
        if "image" not in request.FILES:  # Periksa apakah file ada dalam request
            return JsonResponse({"error": "Tidak ada gambar yang diupload"}, status=400)

        try:
            # Simpan gambar sementara
            file = request.FILES["image"]
            file_path = default_storage.save("temp.jpg", file)

            # Baca dan preprocess gambar
            img = image.load_img(file_path, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # Daftar kelas ulos
            class_labels = ["Pinuncaan", "Ragi Hidup", "Ragi Hotang", "Sadum", "Sibolang", "Tumtuman"]

            # Prediksi menggunakan model
            prediksi = model.predict(img_array)
            prediksi_kelas = class_labels[np.argmax(prediksi)]
            confidence_score = np.max(prediksi[0])

            # akurasi (%)
            akurasi = round(confidence_score * 100, 2)

            # Ambil data dari database
            ulos_penjelasan = get_object_or_404(ulos, nama=prediksi_kelas)

            print("Prediksi Probabilitas:", prediksi[0])
            
            print("akurasi: ", akurasi)

            return JsonResponse({"prediksi": prediksi_kelas, "penjelasan": ulos_penjelasan.deskripsi, "akurasi_predict":str(akurasi)})
        
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Hanya menerima request POST"}, status=400)

def index(request):
    return render(request, "klasifikasi_ulos/index.html")
