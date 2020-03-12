from django.urls import path
from . import views


app_name = 'vector_db'

urlpatterns = [
    path('', views.index),
    path('imgTovec/', views.save_vector_ela),
    path('searchVec/', views.search_vector_service),
    path('putdb/', views.put_database)
]
