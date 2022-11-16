const { Given, When, Then } = require('@wdio/cucumber-framework');


//This scenario was to check if the products are being added to the cart or not

Given('I select a {string}', async (product) => {
    await browser.pause(1000);
    const productitem = $("//div[@class = 'card']/div/a/h2[contains(text(), '"+product+"')]");   //gets a product with specific name
    await productitem.click();
    await browser.pause(1000);
});

When('I click on add to Cart button', async () => {
    const btncart = $("//button[contains(text(), 'Add to Cart')]");   //selects the add to cart button 
    await btncart.click();
    await browser.pause(1000);
});


Then('I can see the {string} in cart', async (product) => {

    const cartitem = $("//li/div/div[2]/a[contains(text(), '"+product+"')]");    //selects the cart item name
    await expect(cartitem).toBeExisting();
});