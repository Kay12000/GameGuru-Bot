from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_from_directory
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
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configurar la API de OpenAI
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    import openai
    openai.api_key = api_key
else:
    raise ValueError("No se encontró la clave API de OpenAI en las variables de entorno.")

print(f"Clave API cargada: {api_key}")

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


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['question']
    user_id = request.json['user_id']  # Añade el ID del usuario para rastrear preguntas personalizadas
    response = get_response(user_input, msg, user_id)  # Pasa 'user_id' y 'msg' a 'get_response'
    
    if "estrategia" in user_input.lower():
        return jsonify(response="¿Para qué juego necesitas una estrategia? (e.g., League of Legends, Dota 2, Fortnite)")
    elif "rendimiento" in user_input.lower():
        return jsonify(response="Proporcióname el ID del jugador para obtener el análisis de rendimiento.")
    else:
        return jsonify(response=response)

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
def get_response(question, msg, user_id):
    db_data = read_db()
    user_questions = db_data.get('questions', {}).values()
    for q in user_questions:
        if q['user_id'] == user_id and q['content'].lower() in question.lower():
            print("Pregunta encontrada en db.json")
            return q['answer']
    preprocessed_question = preprocess(question)
    if preprocessed_question in preprocessed_questions:
        print("Pregunta encontrada en training_data.json")
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
            response = get_gpt4o_mini_answer(question, msg)
            # Guardar la respuesta en la base de datos
            db_data['questions'][len(db_data['questions']) + 1] = {'user_id': user_id, 'content': question, 'answer': response}
            write_db(db_data)
            return response + " (La respuesta se ha guardado correctamente)"

@app.route('/get_response', methods=['POST'])  # Agrega esta ruta
def get_response_route():
    data = request.get_json()
    question = data.get('question')
    user_id = data.get('user_id')
    response = get_response(question, msg, user_id)
    return jsonify({'response' : response})

@app.route('/save_response', methods=['POST'])
def save_response():
    global data  # Asegúrate de acceder a la variable global 'data'
    
    # Obtener los datos de la solicitud
    request_data = request.get_json()
    question = request_data.get('question')
    response = request_data.get('response')
    user_id = request_data.get('user_id')

    # Validar y Sanitizar los Datos
    question = question.strip()
    response = response.strip()

    # Guardar en la base de datos
    questions_table.insert({'user_id': user_id, 'content': question, 'answer': response})

    # Actualizar el modelo con la nueva pregunta y respuesta
    if isinstance(data, dict):
        data = list(data.items())
    data.append((question, response))
    questions, answers = zip(*data)
    preprocessed_questions = [preprocess(q) for q in questions]
    model.fit(preprocessed_questions, answers)
    save_training_data()

    return jsonify({"status": "success"})


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

# Función para obtener respuesta de GPT-4o-mini
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
        # ... (otras respuestas no satisfactorias)
    ]
    
    if any(phrase in answer for phrase in non_satisfactory_answers):
        print("No tengo una respuesta para eso. ¿Cuál debería ser la respuesta?")
        return "Lo siento, no tengo información sobre eso."
    elif answer:
        print("Respuesta encontrada en GPT-4o-mini")
        # Guardar la pregunta y respuesta en la lista
        questions_table.append({'user_id': 'system', 'content': query, 'answer': answer})
        return answer
    else:
        print("Respuesta no encontrada en GPT-4o-mini")
        return "Lo siento, no tengo información sobre eso."

# Función para guardar datos de entrenamiento
def save_training_data():
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

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
     
    