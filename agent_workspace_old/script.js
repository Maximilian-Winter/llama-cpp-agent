document.getElementById('send-button').addEventListener('click', function() {
    var message = document.getElementById('input-field').value;
    if (message) {
        document.getElementById('messages-container').innerHTML += '<p>' + message + '</p>';
        document.getElementById('input-field').value = '';
    }
});