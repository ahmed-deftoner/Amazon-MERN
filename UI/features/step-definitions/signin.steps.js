const { Given, When, Then } = require('@wdio/cucumber-framework');


Given('I am signed out', async () => {
    browser.url('http://localhost:3000')
    const signinlink = $('=Sign In');
    const dropdown = $('.dropdown');
    const signoutlink = $('=Sign Out');
  
    if(await signinlink.isExisting() == false) {
      await browser.pause(7000);
      await dropdown.click();
      await expect(signoutlink).toBeClickable();
      await signoutlink.click();
    }

    await expect(signinlink).toBeClickable();
});

Given('I open sign in page', function () {
    browser.url('http://localhost:3000/signin')
  });

When('I enter email as {string} and password as {string}', async (string, string2) => {
    const button = $("//button[contains(text(), 'Sign In')]");
    await $('#email').setValue(string);
    await $('#password').setValue(string2);
    await button.click();
});

Then('Profile page has same email as {string}', async (email) => {
    const dropdown = $('.dropdown');
    const link = $('=User Profile');
    const emailfield = $('#email');

    await dropdown.click();
    await link.click();
    await expect(emailfield).toHaveValue(email);
});
