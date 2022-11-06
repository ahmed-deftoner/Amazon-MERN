const { Given, When, Then } = require('@wdio/cucumber-framework');

Given('I open Google url', function () {
    // Write code here that turns the phrase above into concrete actions
    browser.url('https://www.google.com/');
});

When('I search for text {string}', async (string) => {
    await (await $('[name="q"]')).setValue(string);
    browser.keys("\uE007");
});

Then('Results displayed should contain {string}', async (string) => {
    await expect($('//h3[contains(text(), "Testing")]')).toHaveTextContaining(string);
});