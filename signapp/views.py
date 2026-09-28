import tempfile


from django.shortcuts import render
from pathlib import Path
from django.conf import settings
from django.http import Http404, HttpResponse, JsonResponse
import joblib
import pickle
import re
import torch
import numpy as np
import __main__




from transformers import T5Config, T5ForConditionalGeneration

from .lemmatizer import lemmatize_word
from .tense_wrapper import TenseModelWrapper
from .gru_model import GRUClassifier




__main__.TenseModelWrapper = TenseModelWrapper
__main__.GRUClassifier = GRUClassifier








# -----------------------------
# Load Tense Model (cache)
# -----------------------------
_PIPE = None




LABEL_MAP = {
  0: "ভবিষ্যৎ",
  1: "বর্তমান",
  2: "অতীত",
}




def get_pipe():
  global _PIPE




  if _PIPE is None:
      model_path = Path(settings.BASE_DIR) / "ml_models" / "final_test_django_tense_pipeline.pkl"




      if not model_path.exists():
          raise FileNotFoundError(f"Tense model file not found: {model_path}")




      # Correct: load using full absolute path
      _PIPE = joblib.load(model_path)




  return _PIPE

# ── Greedy Word Grouper ──────────────────────────────
def get_word_groups(request):
    words_param = request.GET.get("words", "")
    if not words_param:
        return JsonResponse({"groups": []})

    words = words_param.split(",")
    word_base = Path(settings.BASE_DIR) / "static" / "classSign"

    groups = []
    i = 0

    while i < len(words):
        matched = False

        # সবচেয়ে বড় combination থেকে শুরু করো
        for j in range(len(words), i, -1):
            chunk = words[i:j]
            joined = "_".join(chunk)  # জঙ্গি + বিমান → জঙ্গি_বিমান
            file_path = word_base / f"{joined}.sigml"

            if file_path.exists():
                groups.append({
                    "display": " ".join(chunk),  # UI তে দেখাবে
                    "filename": f"{joined}.sigml", # ফাইল খুঁজবে
                    "mode": "sign"
                })
                i = j
                matched = True
                break

        if not matched:
            # single word, sigml নেই → spell করবে
            groups.append({
                "display": words[i],
                "filename": f"{words[i]}.sigml",
                "mode": "spell"
            })
            i += 1

    return JsonResponse({"groups": groups})

def get_root_words(text):
    words = text.split()
    roots = []

    for w in words:
        try:
            lemma = lemmatize_word(w)
            if lemma:
                # যদি "মানুষ গুলো" আসে → split করে দাও
                roots.extend(lemma.split())
        except Exception as e:
            print("Lemma error:", e)
            roots.append(w)

    return roots


def normalize_word(word):
 word = word.strip()
 word = re.sub(r"[^\u0980-\u09FFA-Za-z0-9]", "", word)
 return word




# -----------------------------
# Home Page
# -----------------------------
def home(request):
  return render(request, "home.html")



# -----------------------------
# Helper: Predict tense safely
# -----------------------------
def predict_tense_label(model, text):
    pred = model.predict([text])

    if isinstance(pred, (list, tuple, np.ndarray)):
        pred_value = pred[0]
    else:
        pred_value = pred

    if isinstance(pred_value, np.generic):
        pred_value = pred_value.item()

    # ✅ Bangla mapping
    if isinstance(pred_value, (int, np.integer)):
        return LABEL_MAP.get(pred_value, str(pred_value))

    pred_str = str(pred_value).strip()

    return {
        "Present": "বর্তমান",
        "Past": "অতীত",
        "Future": "ভবিষ্যৎ",
    }.get(pred_str, pred_str)


# -----------------------------
# Tense Detect + Root Word API
# -----------------------------
def detect_tense(request):
    text = request.GET.get("text", "").strip()

    if not text:
        return JsonResponse({"words": [], "tense": ""})

    try:
        model = get_pipe()
        tense_bn = predict_tense_label(model, text)   # ✅ Bangla

        root_words = get_root_words(text)

        # 🔥 important
        root_words.append(tense_bn)

        return JsonResponse({
            "words": root_words,
            "tense": tense_bn
        })

    except Exception as e:
        print("detect_tense error:", e)
        return JsonResponse({
            "error": str(e),
            "words": [],
            "tense": ""
        }, status=500)




# -----------------------------
# SigML file loader
# -----------------------------
def sigml_file(request, filename, mode=None):
  word_base = Path(settings.BASE_DIR) / "static" / "classSign"
  char_base = Path(settings.BASE_DIR) / "static" / "classSignChars"




  if mode == "word":
      path = word_base / filename
  elif mode == "char":
      path = char_base / filename
  else:
      word_path = word_base / filename
      char_path = char_base / filename




      if word_path.exists():
          path = word_path
      elif char_path.exists():
          path = char_path
      else:
          raise Http404("SIGML not found")




  if not path.exists():
      raise Http404("SIGML not found")




  body = path.read_text(encoding="utf-8", errors="ignore").strip()
  low = body.lower()




  if "<sigml" not in low:
      body = f'<?xml version="1.0" encoding="utf-8"?>\n<sigml>\n{body}\n</sigml>\n'




  return HttpResponse(body, content_type="text/xml; charset=utf-8")


import torch
import pickle
from pydub import AudioSegment
from pathlib import Path
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import librosa
import numpy as np
import tempfile


# Global variables for model and label encoder
_EMOTION_MODEL = None
_EMOTION_LABEL_ENCODER = None


# Load the emotion detection model
def load_emotion_model():
   global _EMOTION_MODEL, _EMOTION_LABEL_ENCODER


   if _EMOTION_MODEL is not None and _EMOTION_LABEL_ENCODER is not None:
       return _EMOTION_MODEL, _EMOTION_LABEL_ENCODER


   try:
       # Load the emotion model and label encoder
       emotion_model_path = Path(settings.BASE_DIR) / "ml_models" / "emotion_model_all_emo_django.pkl"
       with open(emotion_model_path, "rb") as f:
           data = pickle.load(f)


       _EMOTION_MODEL = CNN_LSTM(num_classes=len(data['label_encoder'].classes_))
       _EMOTION_MODEL.load_state_dict(data['model_state_dict'])
       _EMOTION_MODEL.eval()


       _EMOTION_LABEL_ENCODER = data['label_encoder']
       return _EMOTION_MODEL, _EMOTION_LABEL_ENCODER


   except Exception as e:
       print(f"Error loading emotion model: {str(e)}")
       return None, None


# Feature extraction for emotion model
def extract_features(file_path):
   y, sr = librosa.load(file_path, duration=3)
   S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
   S_db = librosa.power_to_db(S, ref=np.max)
   mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)


   features = np.vstack([S_db, mfccs])
   features = (features - np.mean(features)) / (np.std(features) + 1e-6)


   return features


# Emotion detection function (same as before)
def detect_emotion_from_file(file_path):
   feat = extract_features(file_path)


   feat = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


   with torch.no_grad():
       out = _EMOTION_MODEL(feat)
       pred = torch.argmax(out, 1).item()


   return _EMOTION_LABEL_ENCODER.inverse_transform([pred])[0]


# Emotion detection API
@csrf_exempt
def detect_emotion(request):
   if request.method != "POST":
       return JsonResponse({"error": "Only POST method is allowed"}, status=405)


   if "audio" not in request.FILES:
       return JsonResponse({"error": "No audio file uploaded"}, status=400)


   temp_path = None
   try:
       audio_file = request.FILES["audio"]
       suffix = Path(audio_file.name).suffix if Path(audio_file.name).suffix else ".webm"


       with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
           for chunk in audio_file.chunks():
               temp_audio.write(chunk)
           temp_path = temp_audio.name


       # WebM → wav conversion
       wav_path = temp_path.replace(".webm", ".wav")
       audio = AudioSegment.from_file(temp_path)
       audio.export(wav_path, format="wav")


       # Ensure the model is loaded correctly
       model, _ = load_emotion_model()


       if model is None:
           return JsonResponse({"error": "Emotion model is not loaded properly"}, status=500)


       emotion = detect_emotion_from_file(wav_path)
       return JsonResponse({"emotion": emotion})


   except Exception as e:
       print("detect_emotion error:", e)
       return JsonResponse({"error": f"Emotion detection failed: {str(e)}"}, status=500)


import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
   def __init__(self, num_classes=5):
       super().__init__()
       self.cnn = nn.Sequential(
           nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
           nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
           nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
       )
       self.lstm = nn.LSTM(input_size=(64 + 40) // 8 * 128, hidden_size=256,
                           batch_first=True, bidirectional=True)
       self.dropout = nn.Dropout(0.3)
       self.fc = nn.Linear(512, num_classes)


   def forward(self, x):
       b = x.size(0)
       x = self.cnn(x)
       x = x.permute(0, 3, 1, 2).reshape(b, x.size(3), -1)
       _, (hn, _) = self.lstm(x)
       hn = torch.cat((hn[-2], hn[-1]), dim=1)
       return self.fc(hn)

