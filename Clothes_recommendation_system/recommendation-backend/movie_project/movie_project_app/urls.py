from django.urls import path, include
from django.contrib import admin
from movie_project_app import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("movies/", views.MovieView.as_view(), name="movie-list"),

]
