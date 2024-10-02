from django.urls import path, include
import debug_toolbar
from django.contrib import admin
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
#from movieProject.views import CreateUserView

urlpatterns = [
    path("admin/", admin.site.urls),
    #path("api/user/register", CreateUserView.as_view(), name="register"),
    path("api/token", TokenObtainPairView.as_view(), name="get_token"),
    path("api/token/refresh/", TokenRefreshView.as_view(), name="refresh"),
    path("api-auth", include("rest_framework.urls")),
    path("api/", include("movieProjectt.urls")),
    path("__debug__/", include(debug_toolbar.urls))
]
