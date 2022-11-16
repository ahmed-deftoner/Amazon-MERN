Feature: Add prdouct

    add product to cart
    Background: Check user signed in
    Given I am signed in

    Scenario Outline: product added to cart
        Given I select a "<product>"
        When I click on add to Cart button
        Then I can see the "<product>" in cart

        Examples:
            | product         |
            | Adidas Fit Pant |
            | Puma Slim Pant  |