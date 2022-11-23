const { Given, When, Then } = require('@wdio/cucumber-framework');

When('I select a {string} and post a {string}', async (rating, comment) => {
    const ratingdropdown = $("#rating");
    const commentfield = $("#comment");
    const submitbtn = $("//button[contains(text(), 'Submit')]");

    await ratingdropdown.selectByVisibleText(rating);
    await commentfield.setValue(comment);
    await browser.pause(500);
    await submitbtn.click();
    await browser.pause(1000);
});



Then('I should see the {string} posted or i should see {string}', async (comment, message) => {
    const alertmsg = $(".alert");

    const postedreview = $("//ul/li/p[contains(text(), '"+comment+"')]");
    if((await alertmsg).isExisting() == false){
        await expect(postedreview).toBeExisting();
    }

});