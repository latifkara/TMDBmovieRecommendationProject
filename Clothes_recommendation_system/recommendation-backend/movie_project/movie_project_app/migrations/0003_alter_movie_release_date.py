# Generated by Django 5.0.6 on 2024-07-10 11:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("movie_project_app", "0002_movie_active"),
    ]

    operations = [
        migrations.AlterField(
            model_name="movie",
            name="release_date",
            field=models.DateField(default="2024-01-01"),
        ),
    ]
