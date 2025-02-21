document.getElementById('send-button').addEventListener('click', function() {
    const userInput = document.getElementById('user-input').value;
    if (userInput.trim() === '') return;

    // Mostrar mensaje del usuario inmediatamente
    addMessage(userInput, 'user');

    // Respuesta del asistente después de un pequeño retraso
    getAssistantResponse(userInput).then(assistantResponse => {
        // Mostrar la respuesta del asistente después de un retraso
        setTimeout(() => {
            addMessage(assistantResponse, 'assistant');
        }, 500); // Ajustar el tiempo (500 ms en este caso)
    });

    // Limpiar el campo de entrada
    document.getElementById('user-input').value = '';
});

// Función para mostrar el mensaje letra por letra, NO LO TOQUEEEEN
function addMessage(message, sender) {
    const chatBox = document.getElementById('chat-box');
    const messageElement = document.createElement('div');
    messageElement.classList.add('chat-message', sender);
    messageElement.style.opacity = 0; // Comienza invisible
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight; // Desplazar hacia abajo

    // Efecto de escritura letra por letra solo para el asistente
    if (sender === 'assistant') {
        typeWriter(message, messageElement, 40); // 40 ms de retraso entre letras
    } else {
        messageElement.innerHTML = message; // Mostrar el mensaje completo para el usuario
        messageElement.style.opacity = 1; // Hacer visible el texto al final
    }
}

// Función para el efecto de escritura
function typeWriter(text, element, delay) {
    let index = 0;

    function type() {
        if (index < text.length) {
            element.innerHTML += text.charAt(index);
            index++;
            setTimeout(type, delay);
        } else {
            element.style.opacity = 1; // Hacer visible el texto al final
        }
    }

    type();
}

// AQUI DEBEN AGREGAR AL ASISTENTE PARA QUE RESPONDA LAS PREGUNTAS QUE HARA EL USUARIO
async function getAssistantResponse(input) {
    try {
        const response = await fetch('/get_response', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: input }) // Suponiendo que cada usuario tiene un ID único
        });
        const data = await response.json();
        return data.response;
    } catch (error) {
        console.error('Error al obtener la respuesta del asistente:', error);
        return "Lo siento, ocurrió un error al obtener la respuesta.";
    }
}

// Agregar evento para enviar mensaje con la tecla Enter
document.getElementById('user-input').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault(); // Evitar el comportamiento por defecto de Enter
        document.getElementById('send-button').click(); // Simular clic en el botón de enviar
    }
});

