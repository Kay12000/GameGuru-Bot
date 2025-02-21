from openai import OpenAI

client = OpenAI(api_key="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" )

msg = []

while True:
    userInput = input("Tu: ")
    
    if userInput.lower() in ["adios", "chao", "bye", "hasta luego", "exit", "quit"]:	
        print("Asistente: ¡Hasta luego!")
        break
    
    msg.append({"role": "user", "content": userInput})
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        messages=msg,
        max_completion_tokens=200
    )

    respuesta = completion.choices[0].message.content
    print("Asistente:" + respuesta)
    msg.append({"role": "assistant", "content": respuesta})