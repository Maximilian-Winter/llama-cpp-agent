const chatbox = document.getElementById('chatbox');
const messages = document.getElementById('messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

const API_URL = 'https://api.example.com/chatbot'; // Replace this with the actual API URL

function sendMessage() {
    const text = userInput.value;
    if (!text || text.trim().length === 0) return;

    userInput.value = '';

    const messageElement = createMessageElement('outgoing', text);
    messages.appendChild(messageElement);
    messages.scrollTop = messages.scrollHeight;

    fetch(API_URL, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
    })   .then(response => response.json())   .then(data => {
        const messageElement = createMessageElement('incoming', data.text);
        messages.appendChild(messageElement);
        messages.scrollTop = messages.scrollHeight;
    })   .catch(error => {
        console.error('Error:', error);

        const messageElement = createMessageElement('incoming', 'An error occurred while processing your request. Please try again later.');
        messages.appendChild(messageElement);
        messages.scrollTop = messages.scrollHeight;
    });
}

function createMessageElement(type, text) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', type);
    messageElement.innerText = text;

    return messageElement;
}

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keydown', event => {
    if (event.key === 'Enter') {
        sendMessage();
    }
});