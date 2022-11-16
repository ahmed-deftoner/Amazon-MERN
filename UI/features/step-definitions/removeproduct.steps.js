const { Given, When, Then } = require('@wdio/cucumber-framework');

//First add the product in cart then checks if the product can be removed from cart
Given('I have a {string} in cart', async (product) => {
    await browser.pause(1000);
    const productitem = $("//div[@class = 'card']/div/a/h2[contains(text(), '"+product+"')]");   //gets a product with specific name
    await productitem.click();
    await browser.pause(1000);

    const btncart = $("//button[contains(text(), 'Add to Cart')]");   //selects the add to cart button 
    await btncart.click();
    await browser.pause(1000);

    const cartitem = $("//li/div/div[2]/a[contains(text(), '"+product+"')]");    //selects the cart item name
    await expect(cartitem).toBeExisting();
    await browser.pause(1000);
});

When('I click on remove button', async () => {
    const removebtn = $("//li/div/div[5]/button");
    await removebtn.click();
    await browser.pause(1000);
});

Then('I {string} is removed from cart', async (product) => {
    const cartitem = $("//li/div/div[2]/a[contains(text(), '"+product+"')]");    //selects the cart item name
    await expect(cartitem).not.toBeExisting();   //expects the item to not to be existing
})