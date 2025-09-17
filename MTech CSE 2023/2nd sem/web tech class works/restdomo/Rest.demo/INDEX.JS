const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
 
app = express();
const routes = require('./routes');
app.use(cors());
app.use(bodyParser.json());
 
app.use('/api', routes);
app.listen(3000, () => {
    console.log('Server is running on http://localhost:3000');
});``