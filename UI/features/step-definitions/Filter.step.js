const { Given, When, Then } = require('@wdio/cucumber-framework');

const FilterPage = require('../utility/filterprices.js');


When('I select a range from {string}', async (range) => {
    const rangebtn = $("//div[@class = 'col-1']/div/ul/li/a[contains(text(), '"+range+"')]");
    await rangebtn.click();
    await browser.pause(2000);
});

Then('I see products in same {string}', async (range) => {  //Checks if the products are realy in the range specified or there is no product
    const noproduct = $(".alert");
    if(await noproduct.isExisting() == false){
        const productspage =  $("//div[@class = 'col-3']/div[1]");
        const cards = productspage.$$("//div[@class = 'card']");
        let price;
        await cards[0].$("//div/div[@class = 'row']/div[@class = 'price']").getText().then(  //gets price of first product 
            function(text) {
                price = text;
            }
        );
        let num = FilterPage.pricerange( range,price); //returns true if price in range and returns false if price not in range
        await expect(num).toBeTruthy();
    }

});