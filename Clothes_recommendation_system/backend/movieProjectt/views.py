from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

def RecommendationView(request):
    return render(request, "hello.html", {'name': 'latif'})

def counter(request):
    words = request.POST['text']
    amount_of_words = len(words.split())
    return render(request, 'counter.html', {"wordsLength": amount_of_words})