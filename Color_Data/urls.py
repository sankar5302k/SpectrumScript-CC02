from django.urls import path
from . import views



urlpatterns = [
    path('', views.Data, name='Color_Data'),
]

