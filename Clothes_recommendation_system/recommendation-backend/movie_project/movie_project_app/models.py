from django.db import models

# Create your models here.
class Movie(models.Model):
    movieId = models.IntegerField(null=True, blank=True)
    userId = models.IntegerField(null=True, blank=True)
    gender = models.CharField(max_length=255, null=True, blank=True)
    age = models.IntegerField(null=True, blank=True)
    occupation = models.IntegerField(null=True, blank=True)
    rating = models.IntegerField(null=True, blank=True)
    title = models.CharField(max_length=255, null=True, blank=True)
    genres = models.CharField(max_length=255, null=True, blank=True)
    poster_path = models.CharField(max_length=255, null=True, blank=True)
    active = models.BooleanField(default=False)

    def __str__(self):
        return self.title