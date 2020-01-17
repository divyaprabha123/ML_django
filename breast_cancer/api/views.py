#from django.shortcuts import render
#
## Create your views here.
#
#def home(request):
#    return render(request, 'api/home.html')

from django.shortcuts import render, redirect
from django.views.generic import TemplateView, ListView, CreateView
from django.core.files.storage import FileSystemStorage
from django.urls import reverse_lazy
import pandas as pd
from .functions import breast_cancer, load_model
from sklearn.ensemble import RandomForestClassifier

class Home(TemplateView):
    template_name = 'api/home.html'
    
    
def upload(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES["document"]
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        url = fs.url(name)
        
        data = pd.read_csv("."+url)
        print(data)
        model = load_model('./api/breast_cancer.sav')
        model.n_jobs = 1
        data["target"] = model.predict(data)
        data.to_csv("."+url)
        context['url'] = fs.url(name)
        
    return render(request,'api/upload.html', context)