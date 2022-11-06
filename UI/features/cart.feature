Feature: Check cart functionality

    checks different scenario for cart functionalities
    Background: Check user signed in
    Given I am signed in

    Scenario: cart empty 
        Given I click on cart button
        When I do not have any item and click on procced button
        Then page is not procced