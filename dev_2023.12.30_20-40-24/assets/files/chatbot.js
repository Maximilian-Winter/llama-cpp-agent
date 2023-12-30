const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const messagesDiv = document.getElementById('messages');

let messages = [];

chatForm.addEventListener('submit', (event) => {
    event.preventDefault();

    const newMessage = {
        role: 'user',
        content: userInput.value
    };

    messages.push(newMessage);
    renderMessages();
    userInput.value = '';

    // Send the message to the fake backend for processing
    fetch('/api/process-message', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(newMessage)
    })
    .then((response) => response.json())
    .then((data) => {
        if (data.error) {
            alert(data.error);
        } else {
            const aiMessage = {
                role: 'assistant',
                content: data.response
            };
            messages.push(aiMessage);
            renderMessages();
        }
    })
    .catch((error) => console.error(error));
});

const renderMessages = () => {
    messagesDiv.innerHTML = '';
    for (const message of messages) {
        const messageElement = document.createElement('div');
        messageElement.classList.add(`${message.role}-message`);
        messageElement.innerText = message.content;
        messagesDiv.appendChild(messageElement);
    }
};

renderMessages();