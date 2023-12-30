// scripts.js

const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const conversation = document.getElementById('conversation');

let chatBotResponses = [
    'Hello! How can I assist you today?',
    'I am here to help with anything you need.',
    'What can I do for you?'
];

let currentIndex = 0;

sendButton.addEventListener('click', () => {
    const userMessage = userInput.value.trim();

    if (userMessage !== '') {
        displayUserMessage(userMessage);
        generateChatBotResponse(userMessage);
        userInput.value = '';
    }
});

function displayUserMessage(message) {
    const newMessage = `<div class="user-message">${message}</div>`;
    conversation.innerHTML += newMessage;
}

function generateChatBotResponse(userInputText) {
    const newResponse = `<div class="chatbot-response">${chatBotResponses[currentIndex]}</div>`;
    conversation.innerHTML += newResponse;

    currentIndex = (currentIndex + 1) % chatBotResponses.length;
}