from flask import Flask, request, jsonify
from app.routes import app_routes
from apscheduler.schedulers.background import BackgroundScheduler
from app.models import retrain_model
import os
from dotenv import load_dotenv
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from app.utils import preprocess

# Cargar variables de entorno
if not os.path.exists('.env'):
    print("Advertencia: No se encontró el archivo .env.")
else:
    load_dotenv()

def create_app():
    app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')
    
    # Configuración de la aplicación
    app.config.from_object('app.config.Config')
    
    # Registrar blueprints
    app.register_blueprint(app_routes)

    # Configurar el programador de tareas
    try:
        scheduler = BackgroundScheduler()
        scheduler.add_job(func=retrain_model, trigger="interval", days=1)
        scheduler.start()
        print("Programador iniciado correctamente.")
    except Exception as e:
        print(f"Error al iniciar el programador: {e}")

    return app

if __name__ == '__main__':
    app = create_app()
    
    # Verificar si CUDA está disponible y si cuDNN está habilitado
    print("CUDA disponible:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Dispositivo CUDA:", torch.cuda.get_device_name(0))
        print("Cantidad de GPUs disponibles:", torch.cuda.device_count())
    print("cuDNN habilitado:", torch.backends.cudnn.enabled)

    # Verifica rutas de plantillas y estáticos
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend/templates/index.html')
    if os.path.exists(template_path):
        print(f"Plantilla encontrada en: {template_path}")
    else:
        print(f"Plantilla NO encontrada en: {template_path}")
    
    static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend/static')
    if os.path.exists(static_path):
        print(f"Archivos estáticos encontrados en: {static_path}")
    else:
        print(f"Directorio de archivos estáticos NO encontrado en: {static_path}")

    # Cargar el modelo y el tokenizer entrenados
    try:
        model_folder = 'model_folder'
        tokenizer = DistilBertTokenizer.from_pretrained(model_folder)
        model = DistilBertForSequenceClassification.from_pretrained(model_folder)
        print("Modelo y tokenizer cargados desde", model_folder)
        print("Configuración del modelo:", model.config)
        print(f"El modelo está configurado para {model.config.num_labels} clases.")
        print("Número total de parámetros en el modelo:", sum(p.numel() for p in model.parameters()))
        print("Número de parámetros entrenables:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    except Exception as e:
        print(f"Error al cargar el modelo o tokenizer: {e}")
        exit(1)  # Salir si no se puede cargar el modelo

    app.run(debug=os.getenv('FLASK_DEBUG', 'True').lower() == 'true')  # Ejecuta la aplicación Flask en modo debug
