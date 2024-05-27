from django.urls import path
from . import views

urlpatterns = [
    path('', views.RecommendationView),
    path('counter', views.counter, name='counter')
]

