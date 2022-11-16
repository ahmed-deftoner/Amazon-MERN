Feature: remove product

    removes a product from cart
    Background: Check user signed in
    Given I am signed in
    
    Scenario Outline: product removed from cart
        Given I have a "<product>" in cart
        When I click on remove button
        Then I "<product>" is removed from cart

        Examples:
            | product         |
            | Adidas Fit Pant |