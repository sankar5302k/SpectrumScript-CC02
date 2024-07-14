from django import forms
from .models import images

class imageform(forms.ModelForm):
    class Meta:
        model = images
        fields = ('image',)