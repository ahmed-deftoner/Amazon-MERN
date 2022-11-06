const { Given, When, Then } = require('@wdio/cucumber-framework');

Given('I am on create account page', function () {
    browser.url('http://localhost:3000/register');
});


When('I enter my details {string} , {string} and {string}', async (exname, exemail, expass) => {
    const name = $('#name');
    const email = $('#email');
    const pass = $('#password');
    const conpass = $('#confirmPassword');
    const cnfbtn = $("//button[contains(text(), 'Register')]");


    await name.setValue(exname);
    await email.setValue(exemail);
    await pass.setValue(expass);
    await conpass.setValue(expass);

    await cnfbtn.click();
});


Then('I click register and account is created for {string}  or error message is displayed', async (email) => {

    const msg = $('.alert-danger');
    const dropdown = $('.dropdown');
    const link = $('=User Profile');
    const emailfield = $('#email');
    browser.pause(2000);

    await dropdown.click();
    await link.click();
    await expect(emailfield).toHaveValue(email);

});
