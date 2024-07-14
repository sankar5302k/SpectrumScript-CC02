from django.db import models

class images(models.Model):
    image = models.ImageField(upload_to='images')
    def delete(self):
        self.image.delete()
        super().delete()