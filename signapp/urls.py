from django.urls import path
from django.http import JsonResponse
import json
import os

from .views import home, sigml_file, detect_tense, detect_emotion, get_word_groups


# 🔥 NEW VIEW (for CWASA config)
def cwasa_config(request):
    file_path = os.path.join(os.path.dirname(__file__), '..', 'template', 'cwaclientcfg.json')

    with open(file_path) as f:
        data = json.load(f)

    return JsonResponse(data)


urlpatterns = [
    path("", home, name="home"),
    path("detect/", detect_tense, name="detect_tense"),

    path("sigml/word/<str:filename>", sigml_file, {"mode": "word"}, name="sigml_word"),
    path("sigml/char/<str:filename>", sigml_file, {"mode": "char"}, name="sigml_char"),
    path("sigml/<str:filename>", sigml_file, name="sigml_file"),

    path("detect-emotion/", detect_emotion, name="detect_emotion"),
    path("word-groups/", get_word_groups, name="word_groups"),
    # 🔥 ADD THIS LINE
    path("cwaclientcfg.json", cwasa_config, name="cwasa_config"),
]