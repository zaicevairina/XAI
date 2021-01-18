from django.contrib import admin
from .models import LearningModel


class LearningModelAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'title', 'fields')


admin.site.register(LearningModel, LearningModelAdmin)
# Register your models here.
