
const { faker } = require("@faker-js/faker");
module.exports = {
  generateRandomPayload
};

function generateRandomPayload(userContext, events, done) {    
  var payload = {
    "email":"email",
    "name":"name",
    "password":"password"
  };
  payload.email =  faker.internet.email();
  payload.name = faker.internet.userName()
  payload.password = faker.internet.password()
    
  userContext.vars.payload = payload;  
  return done();
}

 