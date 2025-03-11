import os
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect, LangDetectException
from translate import Translator
import re
from deep_translator import GoogleTranslator

nltk.download('punkt')
nltk.download('stopwords')

# Lista de stopwords en español y palabras importantes a preservar
stop_words = set(stopwords.words('spanish'))
important_words = {"cómo", "qué", "por qué", "cuál", "cuándo", "dónde", "quién", "qué", "cómo"}

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp3', 'wav', 'avi', 'mp4', 'mkv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_db():
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'db.json')
    try:
        if not os.path.exists(db_path):
            # Crear el archivo db.json si no existe
            initial_data = {"questions": {}}
            with open(db_path, 'w', encoding='utf-8') as file:
                json.dump(initial_data, file, ensure_ascii=False, indent=4)
        with open(db_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error al leer la base de datos: {e}")
        return {"questions": {}}  # Fallback: estructura vacía

def write_db(data):
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'db.json')
    try:
        with open(db_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error al escribir en la base de datos: {e}")

def add_question_to_db(question, user_id):
    db_data = read_db()
    questions_db = db_data.get('questions', {})
    
    # Crear un nuevo ID único para la pregunta
    new_id = str(len(questions_db) + 1)
    questions_db[new_id] = {
        "user_id": user_id,
        "content": question,
        "answer": None  # Sin respuesta por ahora
    }
    
    db_data["questions"] = questions_db
    write_db(db_data)

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zñáéíóúü0-9\s]', '', text)  # Permitir caracteres en español
    text = re.sub(r'\s+', ' ', text).strip()  # Espacios adicionales
    return text

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

# Crear una instancia de GoogleTranslator
translator_instance = GoogleTranslator()
SUPPORTED_LANGUAGES = translator_instance.get_supported_languages(as_dict=True)

def translate_text(text, src_lang, dest_lang):
    try:
        if src_lang not in SUPPORTED_LANGUAGES.values():
            print(f"Idioma no soportado: {src_lang}. Usando inglés ('en') como idioma fuente predeterminado.")
            src_lang = 'en'  # Usa inglés como predeterminado si el idioma no es soportado.
        return GoogleTranslator(source=src_lang, target=dest_lang).translate(text)
    except Exception as e:
        print(f"Error al traducir: {e}")
        return text  # Devuelve el texto original si la traducción falla.

