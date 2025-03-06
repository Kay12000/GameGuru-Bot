import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import wikipediaapi
import requests
import sympy as sp
import json
import openai
from datetime import datetime
import os
from tinydb import TinyDB, Query
import threading
import firebase_admin
from firebase_admin import credentials, db


# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Configurar la base de datos con TinyDB
db = TinyDB('db.json')
users_table = db.table('users')
questions_table = db.table('questions')

# Configuración para la subida de archivos
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp3', 'wav', 'avi', 'mp4', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Funciones para leer y escribir db.json
db_lock = threading.Lock()
training_data_lock = threading.Lock()

def read_db():
    with db_lock:  # Asegurar que solo un hilo pueda leer de db.json a la vez
        with open('db.json', 'r', encoding='utf-8') as file:
            return json.load(file)

def write_db(data):
    with db_lock:  # Asegurar que solo un hilo pueda escribir en db.json a la vez
        with open('db.json', 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

# Funciones para leer y escribir training_data.json
def read_training_data():
    with training_data_lock:  # Asegurar que solo un hilo pueda leer de training_data.json a la vez
        with open('training_data.json', 'r', encoding='utf-8') as file:
            return json.load(file)

def write_training_data(data):
    with training_data_lock:  # Asegurar que solo un hilo pueda escribir en training_data.json a la vez
        with open('training_data.json', 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

# Cargar datos de entrenamiento desde un archivo JSON
try:
    with open('training_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    data = []

# Asegúrate de que 'data' sea una lista
data = list(data)

# Separar preguntas y respuestas
questions, answers = zip(*data) if data else ([], [])

# Función para preprocesar texto
stop_words = set(stopwords.words('spanish'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return " ".join(filtered_tokens)

# Preprocesar preguntas
preprocessed_questions = [preprocess(q) for q in questions]

# Crear el modelo de clasificación
print("Entrenando el modelo...")
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(preprocessed_questions, answers)
print("Modelo entrenado.")

# Variable para almacenar mensajes
msg = []

# Función para predecir respuesta
def get_response(question, msg, user_id=None):
    user_questions = []
    if user_id:
        user_questions = questions_table.search(Query().user_id == user_id)
    for q in user_questions:
        if q['content'].lower() in question.lower():
            return q['answer']
    preprocessed_question = preprocess(question)
    if preprocessed_question in preprocessed_questions:
        return model.predict([preprocessed_question])[0]
    else:
        if is_math_question(question):
            return solve_math_question(question)
        if is_date_question(question):
            return get_current_date()
        wikipedia_summary = get_wikipedia_summary(question)
        if wikipedia_summary:
            return wikipedia_summary
        else:
            return get_gpt4o_mini_answer(question, msg)

# Función para determinar si es una pregunta matemática
def is_math_question(question):
    try:
        sp.sympify(question)
        return all(char.isdigit() or char in "+-*/^() " for char in question)
    except (sp.SympifyError, TypeError):
        return False

# Función para determinar si es una pregunta sobre la fecha
def is_date_question(question):
    date_keywords = ["fecha", "día", "hoy"]
    return any(keyword in question.lower() for keyword in date_keywords)

# Función para obtener la fecha actual
def get_current_date():
    return f"Hoy es {datetime.now().strftime('%d de %B de %Y')}."

# Función para resolver preguntas matemáticas
def solve_math_question(question):
    try:
        result = sp.sympify(question)
        return f"La respuesta es: {result}"
    except (sp.SympifyError, TypeError):
        return "Lo siento, no puedo resolver esa operación matemática."

# Función para obtener resumen de Wikipedia
def get_wikipedia_summary(query):
    print(f"Buscando en Wikipedia: {query}")
    wiki_wiki = wikipediaapi.Wikipedia(language='es', user_agent='asistente-estudianti/1.0 (kay.1200000@gmail.com)')
    page = wiki_wiki.page(query)
    if page.exists():
        print("Página encontrada en Wikipedia")
        return page.summary[:500]
    else:
        print("Página no encontrada en Wikipedia")
        return None

# Función para obtener respuesta de GPT-4o-mini
def get_gpt4o_mini_answer(query, msg):
    print(f"Buscando en GPT-4o-mini: {query}")
    client = openai.OpenAI(api_key=openai.api_key)
    
    system_message = {
        "role": "system",
        "content": "Eres GameGuru Bot, un asistente virtual creado por el equipo Mugiwaras para ayudar con información sobre videojuegos electrónicos."
    }
    
    if not msg:
        msg.append(system_message)
        welcome_message = {
            "role": "assistant",
            "content": "¡Hola! Soy GameGuru Bot, tu asistente virtual creado por el equipo Mugiwaras. Estoy aquí para ayudarte con cualquier consulta relacionada con videojuegos electrónicos. ¿En qué puedo ayudarte hoy?"
        }
        msg.append(welcome_message)
    
    msg.append({"role": "user", "content": query})
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        messages=msg,
        max_completion_tokens=4096
    )
    answer = completion.choices[0].message.content.strip()
    
    non_satisfactory_answers = [
        "Como soy una inteligencia artificial, no tengo la capacidad",
        "No tengo información",
        "No tengo acceso",
        "No tengo una respuesta para eso",
        "Lo siento, pero no tengo información",
        "Lo siento, pero no tengo información sobre eso",
        "No tengo información sobre eso",
        "No tengo información sobre ese tema",
        "No tengo información sobre este tema",
        "No tengo información sobre ese juego",
        "No tengo información sobre este juego",
        "No tengo información sobre ese evento",
        "No tengo información sobre este evento",
        "No tengo información sobre ese jugador",
        "No tengo información sobre este jugador",
        "No tengo información sobre esa estrategia",
        "No tengo información sobre esa pregunta",
        "No tengo información sobre esa consulta",
        "No tengo información sobre esa operación",
        "No tengo información sobre esa fecha",
        "No tengo información sobre esa operación matemática",
        "No tengo información sobre esa operación aritmética"
    ]
    
    if any(phrase in answer for phrase in non_satisfactory_answers):
        print("No tengo una respuesta para eso. ¿Cuál debería ser la respuesta?")
        new_answer = input("Guardar respuesta: ")
        data.append((query, new_answer))
        questions, answers = zip(*data)
        preprocessed_questions = [preprocess(q) for q in questions]
        model.fit(preprocessed_questions, answers)
        save_training_data()
        print("Nueva pregunta y respuesta añadidas y modelo reentrenado.")
        msg.append({"role": "assistant", "content": new_answer})
        return new_answer
    elif answer:
        print("Respuesta encontrada en GPT-4o-mini")
        return answer
    else:
        print("Respuesta no encontrada en GPT-4o-mini")
        return "Lo siento, no tengo información sobre eso."

# Función para guardar datos de entrenamiento
def save_training_data():
    with open('training_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f)

# Ejemplo de uso
while True:
    user_input = input("Pregúntame algo: ")
    if user_input.lower() in ["adios", "chao", "bye", "hasta luego", "exit", "quit", "salir"]:
        print("¡Hasta luego!")
        break
    response = get_response(user_input, msg)
    print("Respuesta:", response)
    msg.append({"role": "assistant", "content": response})

    
    
    
    
