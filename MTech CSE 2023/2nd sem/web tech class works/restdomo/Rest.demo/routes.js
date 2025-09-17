const express = require('express');
const routes = express.Router();
 
routes.get('/students', (req, res) => {
    res.send('All Students');
});
 
routes.get('/students/:id', (req, res) => {
    const id = req.params.id;
    res.send('Student with id: ' + id);
});
 
routes.post('/students', (req, res) => {
    console.log(req.body);
    res.send('Student is added');   
});
 
routes.put('/students/:id', (req, res) => {
    const id = req.params.id;
    console.log(req.body);
    res.send('Student with id: ' + id + ' is updated');
});
 
routes.delete('/students/:id', (req, res) => {
    const id = req.params.id;
    res.send('Student with id: ' + id + ' is deleted');
});
 
module.exports = routes;
 