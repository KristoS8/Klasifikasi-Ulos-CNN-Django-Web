from django.db import models

# Create your models here.
class ulos(models.Model):
    nama = models.CharField(max_length=100, unique=True)
    deskripsi = models.TextField()

    def __str__(self):
        return self.nama