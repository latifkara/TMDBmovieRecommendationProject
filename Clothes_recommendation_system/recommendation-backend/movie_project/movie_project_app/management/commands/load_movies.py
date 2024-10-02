import os
from pyspark.sql import SparkSession
from django.core.management.base import BaseCommand
from datetime import datetime
from movie_project_app.models import Movie

class Command(BaseCommand):
    #help = 'Load movies from CSV file using PySpark'
    def handle(self, *args, **kwargs):
        # Initialize SparkSession
        spark_builder = SparkSession.builder.appName("fetch_movies").config("spark.executor.extraJavaOptions", "-Xss4m").config(
            "spark.driver.extraJavaOptions", "-Xss4m").getOrCreate()
        movies_df = spark_builder.read.csv("movie_project_app/dataset/recommendation_csv.csv", header=True,
                                           inferSchema=True)

        # Function to parse date

        def safe_integer(value):
            try:
                return int(value)
            except (ValueError, TypeError):
                return None

        # Iterate over DataFrame rows and save to Django model
        for row in movies_df.collect():
            Movie.objects.create(
                movieId=safe_integer(row['movieId']),
                userId=safe_integer(row['userId']),
                gender=row['gender'] if row['gender'] else 'Unkown Gender',
                age=safe_integer(row['age']),
                occupation=safe_integer(row['occupation']),
                rating=safe_integer(row['rating']),
                title=row['title'] if row['title'] else 'Unknown Title',
                genres=row['genres'] if row['genres'] else 'Unknown',
                poster_path=row['poster_path'] if row['poster_path'] else '',
                active=False
            )

        spark_builder.stop()
        self.stdout.write(self.style.SUCCESS('Successfully loaded movies from CSV'))

