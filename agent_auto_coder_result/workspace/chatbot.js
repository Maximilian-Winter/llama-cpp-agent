const apiUrl = '/api/messages';

function sendMessage() {
  const usernameInput = document.getElementById('username');
  const messageInput = document.getElementById('message');

  if (!usernameInput.value || !messageInput.value) {
    alert('Please enter a username and message.');
    return;
  }

  fetch(apiUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      username: usernameInput.value,
      message: messageInput.value
    })
  })
  .then(response => response.json())
  .then(data => {
    displayMessage(data);
  });

  usernameInput.value = '';
  messageInput.value = '';
}

function displayMessage(messageData) {
  const chatHistory = document.getElementById('chat-history');
  chatHistory.innerHTML += `<p><strong>${messageData.username}:</strong> ${messageData.message}</p>`;
}