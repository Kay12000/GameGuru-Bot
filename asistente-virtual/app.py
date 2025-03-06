from flask import Flask
from app.routes import app_routes
from apscheduler.schedulers.background import BackgroundScheduler
from app.models import retrain_model
import os

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
    import torch
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
    
    app.run(debug=True)  # Ejecuta la aplicación Flask en modo de depuración