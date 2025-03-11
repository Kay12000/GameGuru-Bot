import unittest
import io
import os
import torch
import tempfile
import shutil
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from flask import Flask
from app.routes import app_routes
from app.utils import preprocess, read_db, translate_text
from app.config import Config

# Configuración del modelo y tokenizer
model_folder = os.path.abspath('model_folder')
alternative_model_folder = os.path.abspath('alternative_model_folder')
os.makedirs(alternative_model_folder, exist_ok=True)

# Usar DistilBERT preentrenado
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Cargar datos de la base de datos
db_data = read_db()
questions_db = db_data.get('questions', {})
answers = [q_data['answer'] for q_data in questions_db.values() if q_data['answer'] is not None]
questions = [q_data['content'] for q_data in questions_db.values()]
num_labels = len(set(answers))

# Imprimir el número de preguntas y respuestas únicas
print(f"Preguntas únicas: {len(set(questions))}")
print(f"Respuestas únicas: {len(set(answers))}")

# Configuración del modelo
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

# Evaluación del modelo
def evaluate_accuracy(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds)

# Entrenamiento del modelo
def train_model():
    questions = [preprocess(q_data['content']) for q_data in questions_db.values()]
    answers = [q_data['answer'] for q_data in questions_db.values() if q_data['answer'] is not None]
    
    # Mapear respuestas a índices
    answer_to_index = {answer: idx for idx, answer in enumerate(set(answers))}
    labels = [answer_to_index[answer] for answer in answers]
    
    # Tokenización y construcción del dataset
    inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True, max_length=128)
    labels = torch.tensor(labels)
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)

    # Dividir el dataset en entrenamiento y prueba
    train_size = int(0.8 * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Configuración del optimizador
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    for epoch in range(Config.NUM_EPOCHS):
        total_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Época {epoch+1}/{Config.NUM_EPOCHS}, Pérdida promedio: {avg_loss:.4f}")
        
        # Precisión en el conjunto de prueba
        accuracy = evaluate_accuracy(model, test_loader)
        print(f"Precisión en prueba: {accuracy * 100:.2f}%")
        
        # Guardar el modelo
        temp_dir = tempfile.mkdtemp()
        try:
            model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)
            shutil.move(temp_dir, model_folder)
            print(f"Modelo guardado en: {model_folder}")
        except Exception as e:
            print(f"Error al guardar el modelo: {e}")
            raise

    # Verificar que el modelo se haya guardado correctamente
    assert os.path.exists(os.path.join(model_folder, 'pytorch_model.bin')), "El modelo no se guardó correctamente."

# Clase para pruebas unitarias
class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')
        self.app.config['UPLOAD_FOLDER'] = 'uploads'
        self.app.register_blueprint(app_routes)
        self.client = self.app.test_client()

    def test_index(self):
        response = self.client.get('/')
        if response.status_code != 200:
            print("Error al cargar la página principal:", response.data.decode('utf-8'))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<!DOCTYPE html>", response.data)

    def test_upload_file(self):
        data = {'file': (io.BytesIO(b"test data"), 'test.txt')}
        response = self.client.post('/upload', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)

    def test_get_response(self):
        data = {'question': '¿Cuál es mi anime favorito?', 'user_id': '12345'}
        response = self.client.post('/get_response', json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('response', response.json)

    def test_train_model(self):
        # Llama a la función de entrenamiento
        train_model()
        model_saved = os.path.exists(os.path.join(model_folder, 'pytorch_model.bin'))
        self.assertTrue(model_saved, "El modelo no se guardó correctamente.")
        
        # Verifica si la carpeta del modelo existe y contiene los archivos necesarios
        self.assertTrue(os.path.exists(model_folder), "La carpeta 'model_folder' no fue creada.")
        self.assertTrue(os.listdir(model_folder), "La carpeta 'model_folder' está vacía.")
        self.assertTrue(os.path.exists(os.path.join(model_folder, 'pytorch_model.bin')), "El modelo no fue guardado.")
        self.assertTrue(os.path.exists(os.path.join(model_folder, 'config.json')), "El archivo de configuración no fue guardado.")

    def test_preprocess(self):
        text = "Hello, world!"
        self.assertEqual(preprocess(text), "hello world")

    def test_translate_text(self):
        text = "Hola, ¿cómo estás?"
        translated = translate_text(text, src_lang="es", dest_lang="en")
        self.assertEqual(translated.lower().replace("?", ""), "hello, how are you")

    def test_read_db(self):
        db_data = read_db()
        self.assertIn('questions', db_data)
        self.assertGreater(len(db_data['questions']), 0)

if __name__ == '__main__':
    unittest.main()
