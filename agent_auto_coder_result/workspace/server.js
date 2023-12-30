const express = require('express');
const app = express();
const port = process.env.PORT || 3000;

app.use(express.json());

app.post('/api/messages', (req, res) => {
  const { username, message } = req.body;
  
  // Process the message and send a response
  res.json({
    username: username,
    message: message
  });
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});