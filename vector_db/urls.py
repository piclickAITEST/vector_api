from django.urls import path
from . import views


app_name = 'vector_db'

urlpatterns = [
    path('', views.index),
    path('imgTovec/', views.all_process),
]
