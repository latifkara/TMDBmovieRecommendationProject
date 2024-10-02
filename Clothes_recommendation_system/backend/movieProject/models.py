from django.db import models

class Movie(models.Model):
    title = models.CharField(max_length=255, null=True, blank=True)
    release_date = models.DateField(null=True, blank=True)
    genres = models.CharField(max_length=255, null=True, blank=True)
    popularity = models.FloatField(null=True, blank=True)
    vote_average = models.FloatField(null=True, blank=True)
    vote_count = models.IntegerField(null=True, blank=True)
    overview = models.TextField(null=True, blank=True)
    poster_path = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return self.title