Feature: Filter department

    filter products by department
    Scenario Outline: filter by department
        Given  I select a category from "<category>"
        Then I should see the products in same "<category>"

        Examples:
            | category |
            | Pants    |
            | Shirts   |