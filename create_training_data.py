import json

data = [
    ("Hola", "¡Hola! ¿En qué puedo ayudarte?"),
    ("¿Cómo estás?", "Estoy bien, gracias. ¿Y tú?"),
    ("¿Cuál es tu nombre?", "Soy un asistente virtual creado para ayudarte."),
    ("¿Qué puedes hacer?", "Puedo responder preguntas y ayudarte con información."),
    ("Adiós", "¡Hasta luego!"),
    ("¿Qué día es hoy?", "Lo siento, no tengo la capacidad de saber la fecha actual."),
    ("¿Cuál es la fecha de hoy?", "Lo siento, no tengo la capacidad de saber la fecha actual."),
    ("¿Qué hora es?", "Lo siento, no tengo la capacidad de saber la hora actual."),
    ("¿Cómo te llamas?", "Soy un asistente virtual creado para ayudarte."),
    ("¿Qué puedes hacer por mí?", "Puedo responder preguntas y ayudarte con información."),
    ("Hasta luego", "¡Hasta luego!"),
    ("¿Cuál es tu color favorito?", "No tengo preferencias de color."),
    ("¿Cuál es tu comida favorita?", "No tengo preferencias de comida."),
    ("¿Qué tiempo hace hoy?", "Lo siento, no tengo la capacidad de saber el clima actual."),
]

with open('training_data.json', 'w') as f:
    json.dump(data, f)

print("Archivo training_data.json creado con éxito.")