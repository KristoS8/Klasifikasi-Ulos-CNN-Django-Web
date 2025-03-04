from django.urls import path
from . import views

urlpatterns = [
    path("prediksi/", views.predict_image, name="prediksi"),
    path("", views.index, name="index")
]
