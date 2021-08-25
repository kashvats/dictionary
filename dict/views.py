from django.shortcuts import render
from PyDictionary import PyDictionary
# Create your views here.
def word(request):
    return render(request,'index.html')


def index(request):
    if request.method == 'GET':
        Search = request.GET.get('Search')
        ak = PyDictionary()
        meaning = ak.meaning(Search)
        antonyms = ak.antonym(Search)
        synonyms = ak.synonym(Search)
        translate = ak.translate(Search, language='hi')
        return render(request, 'index2.html', {'mean': meaning['Noun'][0], 'ant': antonyms, 'syn': synonyms,'trans':translate})
