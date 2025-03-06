# Asistente Virtual

Este proyecto es un asistente virtual diseñado para ayudar con información sobre videojuegos electrónicos y esports. Utiliza un modelo de aprendizaje automático basado en DistilBERT para procesar preguntas y proporcionar respuestas relevantes.

## Estructura del Proyecto

```
asistente-virtual
├── app
│   ├── __init__.py          # Inicializa la aplicación Flask
│   ├── routes.py            # Define las rutas de la API
│   ├── models.py            # Carga y evalúa el modelo DistilBERT
│   ├── utils.py             # Funciones auxiliares
│   └── config.py            # Configuraciones centralizadas
├── frontend
│   ├── static               # Archivos estáticos (CSS, JS, imágenes)
│   └── templates
│       └── index.html       # Plantilla principal de la aplicación
├── uploads                  # Directorio para archivos subidos
├── db.json                  # Base de datos de preguntas y respuestas
├── training_data.json       # Datos de entrenamiento para el modelo
├── model_folder             # Almacena el modelo DistilBERT entrenado
├── best_model_folder        # Almacena la mejor versión del modelo
├── app.py                   # Punto de entrada de la aplicación
└── README.md                # Documentación del proyecto
```

## Requisitos

- Python 3.x
- Flask
- PyTorch
- Transformers
- NLTK
- OpenAI API

## Instalación

1. Clona el repositorio:
   ```
   git clone <URL_DEL_REPOSITORIO>
   cd asistente-virtual
   ```

2. Crea un entorno virtual y actívalo:
   ```
   python -m venv venv
   source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
   ```

3. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

4. Configura la clave de API de OpenAI en `app/config.py`.

## Uso

1. Ejecuta la aplicación:
   ```
   python app.py
   ```

2. Accede a la aplicación en tu navegador en `http://127.0.0.1:5000`.

## Contribuciones

Las contribuciones son bienvenidas. Si deseas contribuir, por favor abre un issue o envía un pull request.

## Licencia

Este proyecto está bajo la Licencia MIT.