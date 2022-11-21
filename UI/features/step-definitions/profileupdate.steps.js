const { Given, When, Then } = require('@wdio/cucumber-framework');

Given('I am signed in using {string} and {string}', async (email, password) => {
    browser.url('http://localhost:3000/signin');
    const button = $("//button[contains(text(), 'Sign In')]");
    await $('#email').setValue(email);
    await $('#password').setValue(password);
    await button.click();
    await browser.pause(1000);

});

When('I go to profile page', async () => {
    const dropdown = $('.dropdown');
    const link = $('=User Profile');
    const emailfield = $('#email');

    await dropdown.click();
    await link.click();
    await browser.pause(1000);
});

When('I enter {string} , {string} and {string}', async (name, newemail, newpassword) => {
    const namefield = $("#name");
    const emailfield = $("#email");
    const passfield = $("#password");
    const cnfpass = $("#confirmPassword");
    const updatebtn = $("//button[contains(text(), 'Update')]");

    await namefield.setValue(name);
    await emailfield.setValue(newemail);
    await passfield.setValue(newpassword);
    await cnfpass.setValue(newpassword);

    await updatebtn.click();
    await browser.pause(1500);  
});

Then('I should see profile updated', async () => {
    const msgfield = $(".alert");
    if((await msgfield).isExisting()){
        await expect(msgfield).toHaveText("Profile Updated Successfully");  
    }
});