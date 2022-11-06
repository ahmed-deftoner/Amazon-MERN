const { Given, When, Then } = require('@wdio/cucumber-framework');


Given('I am signed in', async () => {
    browser.url('http://localhost:3000/signin');
    const button = $("//button[contains(text(), 'Sign In')]");
    await $('#email').setValue("abd.tahir1122@gmail.com");
    await $('#password').setValue("1122");
    await button.click();

});

Given('I click on cart button', async () => {
    const cartlink = $("=Cart");
    if(await cartlink.isExisting()){
        await cartlink.click();
        await browser.pause(2000);
    }
});


When('I do not have any item and click on procced button', async () => {


});

Then('page is not procced', async () => {
    const msg = $(".alert");
    const button = $("//button[contains(text(), 'Proceed to Checkout')]");

    await expect(msg).toBeExisting();
    await expect(button).not.toBeClickable()

});