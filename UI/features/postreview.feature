Feature: post review

    test post a product review 
    Background: Check user signed in
    Given I am signed in

    Scenario Outline: post a review
        Given I select a "<product>"
        When I select a "<rating>" and post a "<review>"
        Then I should see the "<review>" posted or i should see "<message>"

        Examples:
            | product           | rating        | review           | message                        |
            | Nike Slim Pant    | 5- Excelent   | great fabric     | You already submitted a review |