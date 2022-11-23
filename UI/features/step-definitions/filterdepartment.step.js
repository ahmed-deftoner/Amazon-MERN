const { Given, When, Then } = require('@wdio/cucumber-framework');

Given('I select a category from {string}', async (category) => {
    const sidebar = $("//header/div/button[@class = 'open-sidebar']");
    const categorybtn = $("//aside/ul/li/a[contains(text(), '"+category+"')]"); //clicks on the topleft side bar
    browser.url('http://localhost:3000');
    await sidebar.click();
    await browser.pause(1500);
    await categorybtn.click();
    await browser.pause(1500);
});

Then('I should see the products in same {string}', async (category) => {
    const noproduct = $(".alert");

    if(await noproduct.isExisting() == false){
        const productspage =  $("//div[@class = 'col-3']/div[1]");
        const cards = productspage.$$("//div[@class = 'card']");  //selects all products displayeds

        await cards.forEach(async (x,i) => {   //checks for each product displayed that there is category present in product name
            const name = await x.$("//div/a/h2");
            await expect(name).toHaveTextContaining(category.slice(0,-1));
        })

    }
});