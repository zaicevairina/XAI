import os
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseNotAllowed, HttpResponse, HttpResponseRedirect
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from .models import LearningModel
from .forms import LearningModelForm
from django.conf import settings
import mimetypes
import csv
from xai_func import create_model, XAI, XAI_shap_detail
from wsgiref.util import FileWrapper
from django.views.decorators.clickjacking import xframe_options_exempt
ALLOWED_EXTENSIONS = set(['csv'])


@login_required
def get_all_models(request):
    if request.method == 'GET':
        user = request.user
        learning_models = LearningModel.objects.filter(user=user).values("id", 'title')
        return render(request, 'learning_models/get_all.html', {'models': learning_models})


@login_required
# @xframe_options_exempt
@csrf_exempt
def check_model(request, model_id):
    if request.method == 'GET':
        user = request.user
        learning_models = LearningModel.objects.filter(user=user).filter(id=model_id).values('id', 'title', 'fields')
        if len(learning_models) == 0:
            return JsonResponse({'nothing': 'nothing'})
        fields = learning_models[0]['fields']
        return render(request, 'learning_models/check_model.html', {'fields': fields})
    elif request.method == 'POST':
        request_fields = request.POST
        user = request.user
        learning_models = LearningModel.objects.filter(user=user).filter(id=model_id)
        if len(learning_models) == 0:
            return HttpResponseRedirect({'nothing': 'nothing'})

        for learning_model in learning_models:
            fields = learning_model.fields

            path = learning_model.model_file.path
            name = learning_model.model_file.url
        value_fields = []
        for field in fields:
            value = request_fields[field]

            if value.isnumeric():
                value = int(value)
            elif value.replace('.', '').replace('-', '').isnumeric():
                value = float(value)
            value_fields.append(value)

        print(XAI(value_fields, path))

        lime_path = name.split('.')[0]+'_lime.html'
        shap_1_path = name.split('.')[0]+'_shap_1.html'
        shap_2_path = name.split('.')[0]+'_shap_2.html'
        shap_3_path = name.split('.')[0]+'_shap_3.png'

        return render(request, 'learning_models/result.html', {
            'fields': fields,
            'lime': lime_path,
            'shap_1': shap_1_path,
            'shap_2': shap_2_path,
            'shap_3': shap_3_path,
        })

@login_required
# @xframe_options_exempt
# @csrf_exempt
def field_model(request, model_id, field):
    if request.method == 'GET':

        user = request.user
        learning_models = LearningModel.objects.filter(user=user).filter(id=model_id)
        if len(learning_models) == 0:
            return HttpResponseRedirect({'nothing': 'nothing'})

        for learning_model in learning_models:
            fields = learning_model.fields
            path = learning_model.model_file.path
            name = learning_model.model_file.url

        # if len(learning_models) == 0:
        #     return HttpResponseRedirect({'nothing': 'nothing'})


        print(XAI_shap_detail(path, field))

        shap_3_path = name.split('.')[0]+'_shap_4.png'

        return JsonResponse({
            'shap_4': shap_3_path,
        })

@login_required
@csrf_exempt
def download(request):
    if request.method == 'GET':
        form = LearningModelForm()
        return render(request, 'learning_models/add_model.html', {'form': form})
    elif request.method == 'POST':

        form = LearningModelForm(request.POST, request.FILES)
        if form.is_valid() and (request.FILES['model_file'].content_type == 'text/csv' or request.FILES['model_file'].content_type == 'application/vnd.ms-excel'):
            new_model = LearningModel(user=request.user, model_file=form.cleaned_data['model_file'], title=form.cleaned_data['title'])
            new_model.save()
            model_keys = {}
            csv_path = new_model.model_file.name
            pickle_path = csv_path.split('.')[0]+'.pickle'
            csv_fullpath = new_model.model_file.path
            pickle_fullpath = csv_fullpath.replace(csv_path, pickle_path)
            fields = create_model(csv_fullpath, pickle_fullpath)
            new_model.model_file.name = pickle_path
            new_model.fields = fields
            new_model.save()

            return HttpResponseRedirect('/models')
        else:
            return render(request, 'learning_models/add_model.html', {'msg': 'Был получен не csv файл'})

    else:
        HttpResponseNotAllowed(['GET', 'POST'])
