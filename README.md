# Asistente Virtual

Este proyecto es un asistente virtual diseñado para ayudar con información sobre videojuegos electrónicos y esports. Utiliza un modelo de aprendizaje automático basado en DistilBERT para procesar preguntas y proporcionar respuestas relevantes.

## Estructura del Proyecto

```
asistente-virtual
├── app
│   ├── __init__.py          # Inicializa la aplicación Flask
│   ├── routes.py            # Define las rutas de la API
│   ├── models.py            # Carga y evalúa el modelo DistilBERT
│   ├── utils.py             # Funciones auxiliares (traducción, preprocesamiento, etc.)
│   └── config.py            # Configuraciones centralizadas (parámetros, claves de API)
├── frontend
│   ├── static               # Archivos estáticos (CSS, JS, imágenes)
│   └── templates
│       └── index.html       # Plantilla principal de la aplicación
├── uploads                  # Directorio para archivos subidos
├── db.json                  # Base de datos de preguntas y respuestas
├── training_data.json       # Datos de entrenamiento para el modelo
├── model_folder             # Almacena el modelo DistilBERT entrenado
├── best_model_folder        # Almacena la mejor versión del modelo entrenado
├── app.py                   # Punto de entrada de la aplicación
└── README.md                # Documentación del proyecto

```

## Requisitos

- Python 3.x (se recomienda 3.10 o superior, pero no 3.12 debido a compatibilidades con ciertas bibliotecas).
- Flask: Framework para construir la API web.
- PyTorch: Biblioteca de aprendizaje automático para manejar modelos como DistilBERT.
- Transformers: Manejo de modelos preentrenados (como DistilBERT).
- Scikit-learn: Para métricas y evaluaciones.
- Deep Translator: Para la traducción automática de preguntas.
- NLTK: Para preprocesamiento de texto y manejo de stopwords.

## Instalación

1. Clona el repositorio:
   ```
   git clone https://github.com/Kay12000/GameGuru-Bot.git
   cd asistente-virtual
   ```

2. Crea un entorno virtual y actívalo:
   ```
   python -m venv venv
   source venv/bin/activate  # En Linux/Mac
   venv\Scripts\activate     # En Windows

   ```

3. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

4. Configura la clave de API de OpenAI.
   ```
   Asegúrate de tener una clave válida para la API de OpenAI. Configúrala en el archivo .env o directamente en config.py: OPENAI_API_KEY=tu_clave_de_openai
   ```

5. Descarga los recursos de NLTK
   ```
   Este proyecto utiliza recursos específicos de NLTK. Se descargarán automáticamente la primera vez que ejecutes la aplicación. Si deseas descargarlos manualmente, usa el siguiente comando: 
   python -m nltk.downloader punkt stopwords
   ```

## Uso

1. Ejecuta la aplicación:
   ```
   python app.py
   ```

2. Accede a la aplicación en tu navegador en `http://127.0.0.1:5000`.

Migración de Base de Datos (opcional)
JSON a SQLite
Si decides escalar la base de datos de JSON a SQLite, consulta las instrucciones incluidas en la sección de migración del código.

## Contribuciones

Tus contribuciones son bienvenidas. Si deseas colaborar, por favor sigue estos pasos:

Haz un fork del repositorio.

Crea una nueva rama para tu funcionalidad o corrección de errores:

git checkout -b nueva-funcionalidad
Envía un pull request para revisión.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT, lo que significa que puedes usarlo, modificarlo y distribuirlo libremente, siempre y cuando incluyas la licencia original.