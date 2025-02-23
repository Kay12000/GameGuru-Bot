import json
import chardet

# Leer el archivo JSON original
with open('training_data.json', 'rb') as f:
    raw_data = f.read()

# Detectar la codificación
result = chardet.detect(raw_data)
encoding = result['encoding']

# Decodificar los datos utilizando la codificación detectada
data = json.loads(raw_data.decode(encoding))

# Guardar el archivo JSON decodificado
with open('training_data_decoded.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("Datos decodificados y guardados en 'training_data_decoded.json'")

