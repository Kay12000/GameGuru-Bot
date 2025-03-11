from dotenv import load_dotenv
import os

# Cargar variables de entorno desde .env
load_dotenv()

class Config:
    # Entorno y depuración
    ENV = os.getenv('FLASK_ENV', 'development')  # development o production
    DEBUG = ENV == 'development'

    # Configuración de carpetas y extensiones
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')  # Carpeta predeterminada para archivos subidos
    ALLOWED_EXTENSIONS = os.getenv('ALLOWED_EXTENSIONS', 'txt,pdf,png,jpg,jpeg,gif,mp3,wav,avi,mp4,mkv').split(',')  # Extensiones permitidas

    # Clave de API
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        print("Advertencia: OPENAI_API_KEY no está definida en el archivo .env.")

    # Hiperparámetros ajustables
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 5e-5))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 16))
    NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', 25))
    PATIENCE = int(os.getenv('PATIENCE', 3))
