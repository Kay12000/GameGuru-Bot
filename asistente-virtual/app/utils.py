import os
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect
from translate import Translator

# Lista de stopwords en español y palabras importantes a preservar
stop_words = set(stopwords.words('spanish'))
important_words = {"cómo", "qué", "por qué", "cuál", "cuándo", "dónde", "quién", "qué", "cómo"}

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp3', 'wav', 'avi', 'mp4', 'mkv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_db():
    with open('db.json', 'r', encoding='utf-8') as file:
        return json.load(file)

def write_db(data):
    with open('db.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def preprocess(text):
    try:
        tokens = word_tokenize(text.lower())
        filtered_tokens = [token for token in tokens if token.isalnum() and (token not in stop_words or token in important_words)]
        return " ".join(filtered_tokens)
    except LookupError:
        nltk.download('punkt', quiet=True)
        return preprocess(text)

def detect_language(text):
    return detect(text)

def translate_text(text, src_lang, dest_lang):
    translator = Translator(from_lang=src_lang, to_lang=dest_lang)
    translation = translator.translate(text)
    return translation