from .views import download, get_all_models, check_model, field_model
from django.urls import path

urlpatterns = [
    path('download/', download, name='download'),
    path('<int:model_id>/', check_model, name='check_model'),
    path('<int:model_id>/<str:field>', field_model, name='check_model'),
    path('', get_all_models, name='get_all'),
]
