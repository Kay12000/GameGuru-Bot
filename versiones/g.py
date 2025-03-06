from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_from_directory
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import wikipediaapi
import requests
import sympy as sp
import json
import openai
from datetime import datetime
import os
from werkzeug.utils import secure_filename
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Configurar la aplicación Flask
app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')

# Configuración para la subida de archivos
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp3', 'wav', 'avi', 'mp4', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'message': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'message': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_url = url_for('uploaded_file', filename=filename, _external=True)
            return jsonify({'message': f'File {filename} uploaded successfully', 'filepath': file_url}), 200
        else:
            allowed_types = ", ".join(ALLOWED_EXTENSIONS)
            return jsonify({'message': f'File type not allowed. Allowed types are: {allowed_types}'}), 400
    except Exception as e:
        app.logger.error(f"Error en la ruta /upload: {e}")
        return jsonify({"error": "Ocurrió un error interno", "message": str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Funciones para leer y escribir db.json
def read_db():
    with open('db.json', 'r', encoding='utf-8') as file:
        return json.load(file)

def write_db(data):
    with open('db.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Error al renderizar la plantilla: {e}", 500

# Lista de stopwords en español y palabras importantes a preservar
stop_words = set(stopwords.words('spanish'))
important_words = {"cómo", "qué", "por qué", "cuál", "cuándo", "dónde", "quién", "qué", "cómo"}

def preprocess(text):
    try:
        # print("Texto original:", text)  # Registro de depuración
        tokens = word_tokenize(text.lower())
        # print("Tokens:", tokens)  # Registro de depuración
        # Filtrar tokens, pero mantener las palabras clave importantes
        filtered_tokens = [token for token in tokens if token.isalnum() and (token not in stop_words or token in important_words)]
        # print("Tokens filtrados:", filtered_tokens)  # Registro de depuración
        return " ".join(filtered_tokens)
    except LookupError:
        nltk.download('punkt', quiet=True)
        return preprocess(text)

# Cargar datos de entrenamiento desde el archivo JSON
file_path = 'training_data.json'
if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
else:
    data = []

# Asegúrate de que 'data' sea una lista
data = list(data)

# Separar preguntas y respuestas
questions, answers = zip(*data) if data else ([], [])

# Preprocesar preguntas
preprocessed_questions = [preprocess(q) for q in questions]

# Cargar datos de entrenamiento desde db.json
db_data = read_db()
questions_db, answers_db = zip(*[(q['content'], q['answer']) for q in db_data.get('questions', {}).values()])

# Unir datos de training_data.json y db.json
all_questions = preprocessed_questions + [preprocess(q) for q in questions_db]
all_answers = answers + answers_db

# Revisar la cantidad de datos
print(f"Número total de preguntas: {len(all_questions)}")
print(f"Número total de respuestas: {len(all_answers)}")

# Mostrar las primeras 5 preguntas y respuestas
print("Primeras 5 preguntas:", all_questions[:5])
print("Primeras 5 respuestas:", all_answers[:5])

# Definir batch_size para calcular el número de lotes
batch_size = 32  # Ajusta según tus necesidades

# Seleccionar una muestra más pequeña de datos para el entrenamiento
total_questions = len(all_questions)
sample_size = min(500, total_questions)  # Ajusta el valor según tus necesidades

# Seleccionar una muestra del conjunto de datos
indices = np.random.choice(range(total_questions), size=sample_size, replace=False)
all_questions_sample = [all_questions[i] for i in indices]
all_answers_sample = [all_answers[i] for i in indices]

# Vectorizar los datos de la muestra
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(all_questions_sample)

# Dividir los datos en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X_vectorized, all_answers_sample, test_size=0.2, random_state=42)

# Revisar el balance de clases
print(f"Distribución de clases antes de balancear: {Counter(y_train)}")

# Aplicar sobremuestreo a los datos de entrenamiento
oversampler = RandomOverSampler(random_state=42)
X_train_balanced, y_train_balanced = oversampler.fit_resample(X_train, y_train)

print(f"Distribución de clases después de balancear: {Counter(y_train_balanced)}")

# Calcular el número de lotes
num_batches = X_train_balanced.shape[0] // batch_size

# Reducir el número de estimadores y la profundidad máxima
rf_model_balanced = RandomForestClassifier(n_jobs=1, random_state=42, n_estimators=20, max_depth=5)
rf_model_balanced.fit(X_train_balanced, y_train_balanced)

# Evaluar el modelo con los datos de validación
rf_val_score_balanced = rf_model_balanced.score(X_val, y_val)
print(f"Score de validación (RandomForest, datos balanceados): {rf_val_score_balanced}")

# Entrenar el modelo en lotes más pequeños
batch_size = 100  # Ajusta el tamaño del lote según tus necesidades
num_batches = X_train_balanced.shape[0] // batch_size

gb_model_balanced = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_depth=3)

for i in range(num_batches):
    X_batch, y_batch = resample(X_train_balanced, y_train_balanced, n_samples=batch_size, random_state=i)
    gb_model_balanced.fit(X_batch, y_batch)

# Evaluar el modelo con los datos de validación
gb_val_score_balanced = gb_model_balanced.score(X_val, y_val)
print(f"Score de validación (GradientBoosting, datos balanceados por lotes): {gb_val_score_balanced}")

# Vectorizador
vectorizer = CountVectorizer()

# Vectoriza los datos de entrenamiento
X_vectorized = vectorizer.fit_transform(all_questions)

# Dividir los datos en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X_vectorized, all_answers, test_size=0.2, random_state=42)

# Revisar el balance de clases
print(f"Distribución de clases antes de balancear: {Counter(y_train)}")

# Aplicar sobremuestreo a los datos de entrenamiento
oversampler = RandomOverSampler(random_state=42)
X_train_balanced, y_train_balanced = oversampler.fit_resample(X_train, y_train)

print(f"Distribución de clases después de balancear: {Counter(y_train_balanced)}")

# Entrenar el modelo con los datos de entrenamiento balanceados
rf_model_balanced = RandomForestClassifier(n_jobs=1, random_state=42, n_estimators=50, max_depth=10)
rf_model_balanced.fit(X_train_balanced, y_train_balanced)

# Evaluar el modelo con los datos de validación
rf_val_score_balanced = rf_model_balanced.score(X_val, y_val)
print(f"Score de validación (RandomForest, datos balanceados): {rf_val_score_balanced}")

# Entrenar el modelo en lotes
batch_size = 1000
num_batches = X_train_balanced.shape[0] // batch_size

gb_model_balanced = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3)

for i in range(num_batches):
    X_batch, y_batch = resample(X_train_balanced, y_train_balanced, n_samples=batch_size, random_state=i)
    gb_model_balanced.fit(X_batch, y_batch)

# Evaluar el modelo con los datos de validación
gb_val_score_balanced = gb_model_balanced.score(X_val, y_val)
print(f"Score de validación (GradientBoosting, datos balanceados por lotes): {gb_val_score_balanced}")

# Dividir los datos vectorizados en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, all_answers, test_size=0.2, random_state=42)

# Función para mejorar y entrenar los modelos
def improve_and_train_models(X_train, y_train, X_test, y_test, cv_splits):
    import gc
    gc.collect()
    
    # Normalización de datos
    scaler = StandardScaler(with_mean=False)  # Mantiene la escasez del vector
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Ajuste de hiperparámetros con GridSearchCV
    param_grid_rf = {
        'n_estimators': [50, 100],  # Reducir el número de árboles
        'max_depth': [10, None]
    }
    rf_model = RandomForestClassifier(n_jobs=1, random_state=42)
    grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=cv_splits, scoring='accuracy')
    grid_search_rf.fit(X_train_scaled, y_train)
    best_rf_model = grid_search_rf.best_estimator_

    param_grid_gb = {
        'n_estimators': [50, 100],  # Reducir el número de árboles
        'learning_rate': [0.01, 0.1]
    }
    gb_model = GradientBoostingClassifier(random_state=42)
    grid_search_gb = GridSearchCV(gb_model, param_grid_gb, cv=cv_splits, scoring='accuracy')
    grid_search_gb.fit(X_train_scaled, y_train)
    best_gb_model = grid_search_gb.best_estimator_

    # Uso de VotingClassifier para combinar modelos
    voting_clf = VotingClassifier(estimators=[
        ('rf', best_rf_model),
        ('gb', best_gb_model),
        ('nb', MultinomialNB())
    ], voting='soft')
    voting_clf.fit(X_train_scaled, y_train)

    # Evaluación del modelo
    y_pred = voting_clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"VotingClassifier - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    return voting_clf

# Combinar datos de ambos archivos
combined_data = list(zip(preprocessed_questions, answers)) + list(zip(questions_db, answers_db))

# Filtrar solo tuplas de dos elementos (pregunta y respuesta)
combined_data = [(q, a) for q, a in combined_data if isinstance(q, str) and isinstance(a, str)]

# Eliminar duplicados
combined_data = list(set(combined_data))

# Preprocesar y vectorizar los datos combinados sin duplicados
preprocessed_questions = [preprocess(q) for q, _ in combined_data]
answers = [a for _, a in combined_data]
X_vectorized = vectorizer.fit_transform(preprocessed_questions)

# Definir el número de splits para la validación cruzada
cv_splits = 5  # Puedes ajustar este valor según tus necesidades


# Continuar con el proceso de entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, answers, test_size=0.2, random_state=42)
improved_model = improve_and_train_models(X_train, y_train, X_test, y_test, cv_splits)

# Cargar el conjunto de datos
iris = load_iris()
X, y = iris.data, iris.target

# Clasificación binaria: seleccionando solo dos clases para simplificar
X, y = X[y != 2], y[y != 2]

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar y ajustar el modelo de Regresión Logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluar el modelo
score = model.score(X_test, y_test)
print(f"Precisión del Modelo: {score}")

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json['question']
        user_id = request.json['user_id']  # Añade el ID del usuario para rastrear preguntas personalizadas
        
        # Definición de msg
        msg = []

        response = get_response(user_input, msg, user_id)  # Pasa 'user_id' y 'msg' a 'get_response'

        if "estrategia" in user_input.lower():
            return jsonify(response="¿Para qué juego necesitas una estrategia? (e.g., League of Legends, Dota 2, Fortnite)")
        elif "rendimiento" in user_input.lower():
            return jsonify(response="Proporcióname el ID del jugador para obtener el análisis de rendimiento.")
        else:
            return jsonify(response=response)
    except KeyError as e:
        return jsonify({"error": "Clave no encontrada en la solicitud JSON", "message": str(e)}), 400
    except Exception as e:
        # Registro del error para análisis
        app.logger.error(f"Error en la ruta /chat: {e}")
        return jsonify({"error": "Ocurrió un error interno", "message": str(e)}), 500

# Función para guardar datos de entrenamiento
def save_training_data():
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Función para predecir respuesta y aprender de nuevas preguntas

# Definir el modelo básico
model = make_pipeline(CountVectorizer(), RandomForestClassifier(n_jobs=1, random_state=42))

# Función para obtener respuesta del modelo básico
def get_response(question, msg, user_id):
    preprocessed_question = preprocess(question)
    print("Pregunta preprocesada:", preprocessed_question)  # Registro de depuración
    
    # Verificar en training_data.json
    for i, q in enumerate(preprocessed_questions):
        if preprocessed_question == q:
            print("Pregunta encontrada en training_data.json")
            predicted_response = model.predict([preprocessed_question])[0]
            print("Respuesta predicha:", predicted_response)  # Registro de depuración
            return predicted_response
    
    # Verificar en db.json
    db_data = read_db()
    for q in db_data.get('questions', {}).values():
        if q['user_id'] == user_id and q['content'].lower() in question.lower():
            print("Pregunta encontrada en db.json")
            return q['answer']
    
    # Búsqueda en Wikipedia
    wikipedia_summary = get_wikipedia_summary(question)
    if wikipedia_summary:
        print("Respuesta encontrada en Wikipedia")
        return wikipedia_summary

    # Obtener respuesta de GPT-4o-mini con manejo de errores
    gpt4o_mini_answer = get_gpt4o_mini_answer(question, msg)
    if gpt4o_mini_answer:
        # Guardar la respuesta en db.json
        db_data['questions'][len(db_data['questions']) + 1] = {'user_id': user_id, 'content': question, 'answer': gpt4o_mini_answer}
        write_db(db_data)
        return gpt4o_mini_answer + " (La respuesta se ha guardado correctamente en db.json)"
    
    # Utilizar el modelo avanzado como último recurso
    advanced_response = get_advanced_response(question)
    if advanced_response:
        data.append((question, advanced_response))
        save_training_data()
        return advanced_response + " (Respuesta generada por el modelo avanzado y guardada en training_data.json)"
    
    return "Lo siento, no tengo información sobre eso."

# Guardar preguntas y respuestas del usuario en training_data.json cuando OpenAI no funciona
@app.route('/save_user_response', methods=['POST'])
def save_user_response():
    user_data = request.get_json()
    question = user_data.get('question')
    user_response = user_data.get('response')
    
    if question and user_response:
        data.append((question, user_response))
        save_training_data()
        return jsonify({"status": "success", "message": "Respuesta guardada en training_data.json."})
    else:
        return jsonify({"status": "error", "message": "Faltan datos para guardar la respuesta."})

@app.route('/get_response', methods=['POST'])
def get_response_route():
    data = request.get_json()
    question = data.get('question')
    user_id = data.get('user_id')
    
    # Definición de msg
    msg = []
    
    response = get_response(question, msg, user_id)
    
    if response:
        print("Respuesta enviada desde el servidor:", response)  # Registro de depuración
        return jsonify({'response': response})
    else:
        print("No se encontró una respuesta adecuada.")  # Registro de depuración
        return jsonify({'response': None})

# Definir el modelo avanzado
advanced_model = make_pipeline(CountVectorizer(), GradientBoostingClassifier())

# Función para predecir respuesta con lógica avanzada
def get_advanced_response(question):
    preprocessed_question = preprocess(question)
    if preprocessed_question in preprocessed_questions:
        print("Pregunta encontrada en el modelo avanzado")
        return advanced_model.predict([preprocessed_question])[0]
    else:
        print("Pregunta no encontrada en el modelo avanzado")
        return "Lo siento, no tengo información sobre eso."

@app.route('/save_response', methods=['POST'])
def save_response():
    global data  # Asegúrate de acceder a la variable global 'data'
    
    request_data = request.get_json()
    question = request_data.get('question')
    response = request_data.get('response')
    user_id = request_data.get('user_id')

    question = question.strip()
    response = response.strip()

    # Guardar en la base de datos
    db_data = read_db()
    db_data['questions'][len(db_data['questions']) + 1] = {'user_id': user_id, 'content': question, 'answer': response}
    write_db(db_data)

    # Actualizar el modelo con la nueva pregunta y respuesta
    if isinstance(data, dict):
        data = list(data.items())
    data.append((question, response))
    questions, answers = zip(*data)
    preprocessed_questions = [preprocess(q) for q in questions]
    model.fit(preprocessed_questions, answers)
    advanced_model.fit(preprocessed_questions, answers)
    save_training_data()

    return jsonify({"status": "success"})

# Función para obtener respuesta de GPT-4o-mini con manejo de errores
def get_gpt4o_mini_answer(query, msg):
    print(f"Buscando en GPT-4o-mini: {query}")
    client = openai.OpenAI(api_key=openai.api_key)
    
    # Añadir mensaje de sistema para que el asistente se identifique como GameGuru Bot
    system_message = {
        "role": "system",
        "content": "Eres GameGuru Bot, un asistente virtual creado por el equipo Mugiwaras para ayudar con información sobre videojuegos electrónicos."
    }
    
    if not msg:  # Solo añadir el mensaje de sistema si 'msg' está vacío
        msg.append(system_message)
        
        # Añadir mensaje de bienvenida
        welcome_message = {
            "role": "assistant",
            "content": "¡Hola! Soy GameGuru Bot, tu asistente virtual creado por el equipo Mugiwaras. Estoy aquí para ayudarte con cualquier consulta relacionada con videojuegos competitivos o esports. ¿En qué puedo ayudarte hoy?"
        }
        msg.append(welcome_message)
    
    msg.append({"role": "user", "content": query})
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            store=True,
            messages=msg,
            max_completion_tokens=4096
        )
        answer = completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error al obtener respuesta de GPT-4o-mini: {e}")
        return None
    
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
        "Lamentablemente, no tengo información",
        "Lamentablemente, no tengo información sobre eso",
        "Lamentablemente, no tengo información sobre ese tema",
        "Lamentablemente, no tengo información sobre este tema",
        "Lamentablemente, no tengo información sobre ese juego",
        "Lamentablemente, no tengo información sobre este juego",
        "Lamentablemente, no tengo información sobre ese evento",
        "Lamentablemente, no tengo información sobre este evento",
        "Lamentablemente, no tengo información sobre ese jugador",
        # ... (otras respuestas no satisfactorias)
    ]
    
    if any(phrase in answer for phrase in non_satisfactory_answers):
        print("Respuesta no satisfactoria de GPT-4o-mini")
        return None
    elif answer:
        print("Respuesta encontrada en GPT-4o-mini")
        return answer
    else:
        print("Respuesta no encontrada en GPT-4o-mini")
        return None

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

# Define la variable `questions_table` como una lista vacía
questions_table = []

# Función para obtener información en tiempo real de eventos de esports
def get_esports_event_info(event_name):
    api_url = f"https://api.example.com/esports/events/{event_name}"
    response = requests.get(api_url)
    if response.status_code == 200:
        event_info = response.json()
        return event_info
    else:
        return "No se pudo obtener información sobre el evento."

# Función para obtener estadísticas y análisis de partidas
def get_performance_analysis(player_id):
    api_url = f"https://api.example.com/esports/players/{player_id}/performance"
    response = requests.get(api_url)
    if response.status_code == 200:
        performance_analysis = response.json()
        return performance_analysis
    else:
        return "No se pudo obtener el análisis de desempeño."

# Función para obtener estrategias de juego
game_strategies = {
    "League of Legends": "Consejo: Mantén siempre visión en el mapa, usa control wards y comunica con tu equipo.",
    "Valorant": "Consejo: Comunica con tu equipo, usa habilidades de forma coordinada y practica la puntería.",
    "Counter-Strike: Global Offensive": "Consejo: Aprende a usar granadas, conoce los mapas y practica la puntería.",
    "Fortnite": "Consejo: Construye para protegerte, recolecta recursos y mejora tu puntería.",
    "Overwatch": "Consejo: Comunica con tu equipo, conoce las fortalezas de cada héroe y practica en grupo.",
    "Dota 2": "Consejo: Farmea oro y experiencia, comunica con tu equipo y aprende a usar los objetos.",
}

def get_game_strategy(game_name):
    return game_strategies.get(game_name, "No se encontró una estrategia para ese juego.")
    
# Ruta para obtener información de eventos de esports
@app.route('/esports_event_info', methods=['POST'])
def esports_event_info():
    event_name = request.form['event_name']
    event_info = get_esports_event_info(event_name)
    return jsonify({"event_info": event_info})
    
# Ruta para obtener análisis de desempeño de jugadores
@app.route('/performance_analysis', methods=['POST'])
def performance_analysis():
    player_id = request.form['player_id']
    analysis = get_performance_analysis(player_id)
    return jsonify({"analysis": analysis})
    
# Ruta para obtener estrategias de juego
@app.route('/game_strategy', methods=['POST'])
def game_strategy():
    game_name = request.form['game_name']
    strategy = get_game_strategy(game_name)
    return jsonify({"strategy": strategy})

if __name__ == '__main__':
    # Verifica que la ruta de la plantilla sea correcta
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend/templates/index.html')
    if os.path.exists(template_path):
        print(f"Plantilla encontrada en: {template_path}")
    else:
        print(f"Plantilla NO encontrada en: {template_path}")

    # Imprime el directorio de trabajo actual
    print("Directorio de trabajo actual:", os.getcwd())

    app.run(debug=True)