from django.urls import path
from movieProject import views

urlpatterns = [
    path("movies/", views.MovieView.as_view(), name="movie-list"),
    # path("movieProjectt/", include("movieProjectt.urls")),


]

