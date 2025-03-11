from flask import Blueprint, request, jsonify, send_from_directory, render_template, url_for, current_app
from werkzeug.utils import secure_filename
import os
from .utils import allowed_file, read_db, write_db, preprocess, translate_text
from .models import get_response, get_game_strategy, get_performance_analysis, get_esports_event_info

# Crea un blueprint para organizar las rutas de la aplicación
app_routes = Blueprint('app_routes', __name__)

# Ruta principal
@app_routes.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        current_app.logger.error(f"Error al renderizar la plantilla: {e}")
        return f"Error al renderizar la plantilla: {e}", 500

# Ruta para subir archivos
@app_routes.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
            file_url = url_for('app_routes.uploaded_file', filename=filename, _external=True)
            return jsonify({"file_url": file_url}), 200
        else:
            allowed_types = ", ".join(current_app.config['ALLOWED_EXTENSIONS'])
            return jsonify({'message': f'File type not allowed. Allowed types are: {allowed_types}'}), 400
    except Exception as e:
        current_app.logger.error(f"Error en la ruta /upload: {e}")
        return jsonify({"error": "Ocurrió un error interno", "message": str(e)}), 500

# Ruta para descargar archivos subidos
@app_routes.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)

# Ruta para obtener la respuesta del modelo
@app_routes.route('/get_response', methods=['POST'])
def get_response_route():
    data = request.get_json()
    question = data.get('question')
    user_id = data.get('user_id')
    
    # Traducción de la pregunta al inglés si es necesario
    lang = preprocess(question)
    if lang != 'en':
        question = translate_text(question, src_lang=lang, dest_lang='en')
    
    response = get_response(question, user_id)
    
    if response:
        return jsonify({'response': response}), 200
    else:
        return jsonify({'response': "Lo siento, no tengo una respuesta para esa pregunta en este momento."}), 404

# Ruta para guardar preguntas y respuestas
@app_routes.route('/save_response', methods=['POST'])
def save_response_route():
    data = request.get_json()
    question = data.get('question')
    answer = data.get('answer')
    user_id = data.get('user_id')
    
    db_data = read_db()
    questions_db = db_data.get('questions', {})
    new_id = str(len(questions_db) + 1)
    questions_db[new_id] = {
        'user_id': user_id,
        'content': question,
        'answer': answer
    }
    write_db(db_data)
    
    return jsonify({'message': 'Response saved successfully'}), 200

# Ruta para obtener información de eventos de esports
@app_routes.route('/esports_event_info', methods=['POST'])
def esports_event_info():
    event_name = request.form.get('event_name', '')
    if not event_name:
        return jsonify({"error": "El nombre del evento es obligatorio"}), 400
    event_info = get_esports_event_info(event_name)
    return jsonify({"event_info": event_info})

# Ruta para análisis de desempeño de un jugador
@app_routes.route('/performance_analysis', methods=['POST'])
def performance_analysis():
    player_id = request.form.get('player_id', '')
    if not player_id:
        return jsonify({"error": "El ID del jugador es obligatorio"}), 400
    analysis = get_performance_analysis(player_id)
    return jsonify({"analysis": analysis})

# Ruta para estrategia de juegos
@app_routes.route('/game_strategy', methods=['POST'])
def game_strategy():
    game_name = request.form.get('game_name', '')
    if not game_name:
        return jsonify({"error": "El nombre del juego es obligatorio"}), 400
    strategy = get_game_strategy(game_name)
    return jsonify({"strategy": strategy})
