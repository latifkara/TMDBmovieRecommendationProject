# Generated by Django 5.0.6 on 2024-07-03 13:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("movie_project_app", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="movie", name="active", field=models.BooleanField(default=False),
        ),
    ]
