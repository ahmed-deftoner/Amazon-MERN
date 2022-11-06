Feature: Sign up feature

    Sign up on website
    Scenario: Sign up using email 
    Given I am on create account page
    When I enter my details "<name>" , "<email>" and "<password>"
    Then I click register and account is created for "<email>"  or error message is displayed
    
    Examples:
        | name          | email                 | password          |
        | ahm           | ahmed4@gmail.com       | 1122              |