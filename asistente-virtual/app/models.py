import json
import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, recall_score
from .utils import preprocess, read_db, write_db, detect_language, translate_text
from .config import Config

# Cargar el modelo y el tokenizer
model_folder = 'model_folder'

def load_model():
    if os.path.exists(model_folder):
        model = DistilBertForSequenceClassification.from_pretrained(model_folder)
        tokenizer = DistilBertTokenizer.from_pretrained(model_folder)
        print("Modelo cargado desde el almacenamiento.")
    else:
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        print("Modelo creado desde cero.")
    return model, tokenizer

model, tokenizer = load_model()

def train_model():
    db_data = read_db()
    questions_db = db_data.get('questions', {})
    questions = [q_data['content'] for q_data in questions_db.values()]
    answers = [q_data['answer'] for q_data in questions_db.values()]
    
    inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True, max_length=128)
    labels = torch.tensor([answers.index(label) for label in answers])
    
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
    train_size = int(0.8 * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    for epoch in range(Config.NUM_EPOCHS):
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    model.save_pretrained(model_folder)
    tokenizer.save_pretrained(model_folder)
    print("Modelo entrenado y guardado.")

def evaluate_validation_loss(model, validation_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in validation_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(validation_loader)

def evaluate_model(model, test_loader):
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

    return all_labels, all_preds

def calculate_metrics(all_labels, all_preds):
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    return accuracy, f1, recall

def save_model(model):
    model.save_pretrained(model_folder)
    tokenizer.save_pretrained(model_folder)

def get_response(question, user_id):
    # Detectar el idioma de la pregunta
    lang = detect_language(question)
    
    # Traducir la pregunta al inglés si es necesario
    if lang != 'en':
        question = translate_text(question, src_lang=lang, dest_lang='en')
    
    preprocessed_question = preprocess(question)
    inputs = tokenizer(preprocessed_question, return_tensors="pt", padding=True, truncation=True, max_length=128)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities).item()
    
    db_data = read_db()
    questions_db = db_data.get('questions', {})
    for q_id, q_data in questions_db.items():
        if q_data['content'].lower() == question.lower() and q_data['user_id'] == user_id:
            response = q_data['answer']
            break
    else:
        response = "Lo siento, no tengo información sobre eso."
    
    # Traducir la respuesta al idioma original si es necesario
    if lang != 'en':
        response = translate_text(response, src_lang='en', dest_lang=lang)
    
    return response

def get_game_strategy(game_name):
    return f"Estrategia para {game_name}"

def get_performance_analysis(player_id):
    return f"Análisis de desempeño para el jugador {player_id}"

def get_esports_event_info(event_name):
    return f"Información del evento {event_name}"

def retrain_model():
    db_data = read_db()
    questions_db = db_data.get('questions', {})
    questions = [q_data['content'] for q_data in questions_db.values()]
    answers = [q_data['answer'] for q_data in questions_db.values()]
    
    inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True, max_length=128)
    labels = torch.tensor([answers.index(label) for label in answers])
    
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
    train_size = int(0.8 * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        val_loss = evaluate_validation_loss(model, test_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            model.save_pretrained(model_folder)
            tokenizer.save_pretrained(model_folder)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                break