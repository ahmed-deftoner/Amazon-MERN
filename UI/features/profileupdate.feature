Feature: Update Profile

    update profile details
    Background: Check user signed out
    Given I am signed out

    Scenario Outline: profile details update
        Given I am signed in using "<email>" and "<password>"
        When I go to profile page
        And I enter "<name>" , "<newemail>" and "<newpassword>"
        Then I should see profile updated

        Examples:
            | email                         | password    | name             | newemail                      | newpassword   |
            | abd.tahir1122@gmail.com       | 1122        | Abdullah         | abd.tahir1122@gmail.com       | 1122          |


