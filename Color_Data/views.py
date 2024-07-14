from django.http import HttpResponse
from django.template import loader
from .forms import imageform
from .backend import main
from .models import images
from pathlib import Path
import os
from django.shortcuts import render 
from json import dumps


BASE_DIR = Path(__file__).resolve().parent.parent

def main_view(request):
    template = loader.get_template('main.html')
    context = {'form': imageform(), 'v': 0}
    return HttpResponse(template.render(context, request))

def Data(request):
    if request.method == 'POST':
        form = imageform(request.POST, request.FILES)
        if form.is_valid():
            initial_obj = form.save(commit=False)
            initial_obj.save()
            link = initial_obj.image.url
            link = str(BASE_DIR) + '/media' + str(link)
            link = str(link).split('/')
            link = '\\'.join(link)
            image_path = Path(link)
            form.save()
            analyzer = main.ImageColorAnalyzer(image_path)
            color = analyzer.analyze_image()
            template = loader.get_template('index.html')
            os.remove(image_path)
            images.objects.all().delete()
            all_color = color[4]
            percentage = color[7]
            context = {
                'numberoftimes': color[8], 
                'all': all_color, 
                'per': percentage, 
                'acr': color[5], 
                'dp': color[1], 
                'sp': color[3], 
                'sdp': color[2], 
                'mc': color[6]
            }
            return HttpResponse(template.render(context, request))
        else:
            context = {'form': form, 'v': 1}
            template = loader.get_template('main.html')
            return HttpResponse(template.render(context, request))

    template = loader.get_template('main.html')
    context = {'form': imageform(), 'v': 0}
    return HttpResponse(template.render(context, request))
