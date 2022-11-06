Feature: Google search

    Google search
    Scenario: Search text on google search
        Given I open Google url
        When I search for text "Testing"
        Then Results displayed should contain "Testing"