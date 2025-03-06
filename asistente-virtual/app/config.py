from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp3', 'wav', 'avi', 'mp4', 'mkv'}
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Hiperparámetros ajustables
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    PATIENCE = 3

# Verificar y crear la carpeta de uploads si no existe
if not os.path.exists(Config.UPLOAD_FOLDER):
    os.makedirs(Config.UPLOAD_FOLDER)