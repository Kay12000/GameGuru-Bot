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

# Configurar la API de OpenAI
openai.api_key = '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'

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Cargar datos de entrenamiento desde un archivo JSON
try:
    with open('training_data.json', 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    data = [
        ("Hola", "¡Hola! ¿En qué puedo ayudarte?"),
        ("¿Cómo estás?", "Estoy bien, gracias. ¿Y tú?"),
        ("¿Cuál es tu nombre?", "Soy un asistente virtual creado para ayudarte."),
        ("¿Qué puedes hacer?", "Puedo responder preguntas y ayudarte con información."),
        ("Adiós", "¡Hasta luego!"),
        ("¿Qué día es hoy?", "Lo siento, no tengo la capacidad de saber la fecha actual."),
        ("¿Cuál es la fecha de hoy?", "Lo siento, no tengo la capacidad de saber la fecha actual."),
        ("¿Qué hora es?", "Lo siento, no tengo la capacidad de saber la hora actual."),
        ("¿Cómo te llamas?", "Soy un asistente virtual creado para ayudarte."),
        ("¿Qué puedes hacer por mí?", "Puedo responder preguntas y ayudarte con información."),
        ("Hasta luego", "¡Hasta luego!"),
        ("¿Cuál es tu color favorito?", "No tengo preferencias de color."),
        ("¿Cuál es tu comida favorita?", "No tengo preferencias de comida."),
        ("¿Qué tiempo hace hoy?", "Lo siento, no tengo la capacidad de saber el clima actual."),
    ]

# Separar preguntas y respuestas
questions, answers = zip(*data)

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

# Función para predecir respuesta
def get_response(question, msg):
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
        # Intentar evaluar la expresión matemática
        sp.sympify(question)
        # Si la evaluación es exitosa y la pregunta contiene solo números y operadores matemáticos, es una pregunta matemática
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
        return page.summary[:500]  # Limitar a 500 caracteres
    else:
        print("Página no encontrada en Wikipedia")
        return None

# Función para obtener respuesta de GPT-4o-mini
def get_gpt4o_mini_answer(query, msg):
    print(f"Buscando en GPT-4o-mini: {query}")
    client = openai.OpenAI(api_key=openai.api_key)
    msg.append({"role": "user", "content": query})
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        messages=msg,
        max_completion_tokens=200
    )
    answer = completion.choices[0].message.content.strip()
    if "No tengo información" in answer or "No tengo acceso" in answer:
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
    with open('training_data.json', 'w') as f:
        json.dump(data, f)

# Ejemplo de uso
msg = []

while True:
    user_input = input("Pregúntame algo: ")
    if user_input.lower() in ["adios", "chao", "bye", "hasta luego", "exit", "quit", "salir"]:
        print("¡Hasta luego!")
        break
    response = get_response(user_input, msg)
    print("Respuesta:", response)
    msg.append({"role": "assistant", "content": response})