from flask import Flask
from .routes import app_routes
from .models import load_model

def create_app():
    app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')
    
    # Configuración de la aplicación
    app.config.from_object('app.config.Config')
    
    # Registrar blueprints
    app.register_blueprint(app_routes)

    # Cargar el modelo y el tokenizer entrenados
    global model, tokenizer
    model, tokenizer = load_model()

    return app