
from .serializers import MovieSerializer
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import Movie

class MovieView(APIView):
    def get(self, request, *args, **kwargs):
        movies = Movie.objects.all()
        serializer = MovieSerializer(movies, many=True)
        return Response(serializer.data)
