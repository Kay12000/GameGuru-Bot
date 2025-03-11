import os
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect
from translate import Translator
import re

# Lista de stopwords en español y palabras importantes a preservar
stop_words = set(stopwords.words('spanish'))
important_words = {"cómo", "qué", "por qué", "cuál", "cuándo", "dónde", "quién", "qué", "cómo"}

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp3', 'wav', 'avi', 'mp4', 'mkv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_db():
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'db.json')
    if not os.path.exists(db_path):
        # Crear el archivo db.json si no existe
        initial_data = {
            "questions": {
                "1": {
                    "user_id": "12345",
                    "content": "¿Cuál es mi anime favorito?",
                    "answer": "Hai to Gensou no Grimgar"
                },
                "2": {
                    "user_id": "12345",
                    "content": "¿Cuál es mi anime seinen favorito?",
                    "answer": "Monster"
                }
            }
        }
        with open(db_path, 'w', encoding='utf-8') as file:
            json.dump(initial_data, file, ensure_ascii=False, indent=4)
    with open(db_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def write_db(data):
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'db.json')
    with open(db_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def preprocess(text):
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar caracteres no alfanuméricos
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Eliminar espacios adicionales
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def detect_language(text):
    return detect(text)

def translate_text(text, src_lang, dest_lang):
    translator = Translator(from_lang=src_lang, to_lang=dest_lang)
    translation = translator.translate(text)
    return translation