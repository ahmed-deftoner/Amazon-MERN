const { Given, When, Then } = require('@wdio/cucumber-framework');


//Checks that the cart is empty
Given('I am signed in', async () => {
    const signinlink = $("=Sign In");
    const button = $("//button[contains(text(), 'Sign In')]");

    browser.url("http://localhost:3000");
    if(await signinlink.isExisting() ) {
        await browser.pause(1000);
        await signinlink.click();
        await $('#email').setValue("abd.tahir1122@gmail.com");
        await $('#password').setValue("1122");
        await button.click();
    }

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