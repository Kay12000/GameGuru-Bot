from app.utils import translate_text

# Probar la traducción
translated_text = translate_text("Hola, ¿cómo estás?", src_lang="es", dest_lang="en")
print(translated_text)  # Debería imprimir: "Hello, how are you?"
