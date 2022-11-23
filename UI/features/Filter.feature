Feature: Filter products

    Filter products on category page 
    Scenario Outline: filter category products
        Given I select a category from "<category>"
        When I select a range from "<range>"
        Then I see products in same "<range>"

        Examples:
            | category      | range         |
            | Pants         | $1 to $10     |
            | Pants         | $10 to $100   |
            | Pants         | $100 to $1000 |