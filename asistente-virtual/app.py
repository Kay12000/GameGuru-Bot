from flask import Flask, request, jsonify
from app.routes import app_routes
from apscheduler.schedulers.background import BackgroundScheduler
from app.models import retrain_model
import os
from dotenv import load_dotenv
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from app.utils import preprocess

load_dotenv()

def create_app():
    app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')
    
    # Configuración de la aplicación
    app.config.from_object('app.config.Config')
    
    # Registrar blueprints
    app.register_blueprint(app_routes)

    # Configurar el programador de tareas
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=retrain_model, trigger="interval", days=1)
    scheduler.start()

    return app

if __name__ == '__main__':
    app = create_app()
    
    # Verificar si CUDA está disponible y si cuDNN está habilitado
    print("CUDA disponible:", torch.cuda.is_available())  # Debería devolver True si CUDA está disponible
    print("cuDNN habilitado:", torch.backends.cudnn.enabled)  # Debería devolver True si cuDNN está habilitado

    # Verifica que la ruta de la plantilla sea correcta
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend/templates/index.html')
    if os.path.exists(template_path):
        print(f"Plantilla encontrada en: {template_path}")
    else:
        print(f"Plantilla NO encontrada en: {template_path}")

    # Imprime el directorio de trabajo actual
    print("Directorio de trabajo actual:", os.getcwd())
    
    # Cargar el modelo y el tokenizer entrenados
    model_folder = 'model_folder'
    tokenizer = DistilBertTokenizer.from_pretrained(model_folder)
    model = DistilBertForSequenceClassification.from_pretrained(model_folder)
    print("Modelo y tokenizer cargados desde", model_folder)
    
    # Imprimir la configuración del modelo
    print("Configuración del modelo:", model.config)
    print(f"El modelo está configurado para {model.config.num_labels} clases.")

    app.run(debug=True)  # Ejecuta la aplicación Flask en modo de depuración