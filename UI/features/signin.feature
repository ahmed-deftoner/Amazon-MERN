Feature: Sign in page

    check sign in
    Background: Check user signed out
    Given I am signed out

    Scenario Outline: Sign in using credentials
        Given I open sign in page
        When I enter email as "<email>" and password as "<password>"
        Then Profile page has same email as "<email>"

        Examples:
            | email                   | password |
            | abd.tahir1122@gmail.com | 1122     |
            | abd.tahir1122@gmail.com | 112      |
