from grapevine.names import spell_names

ROSTER = ["Sam Torres", "Sam Reyes", "Jasper Lin", "Kate Russo"]


def test_near_misses_of_roster_names_take_the_roster_spelling():
    assert spell_names("San Torres, and that is Jasper Lind.", ROSTER) == "Sam Torres, and that is Jasper Lin."


def test_other_names_and_far_misses_stay_as_heard():
    text = "Kim Russo met Sam Reyes and Andrew Gao at Red Rocks."
    assert spell_names(text, ROSTER) == text


def test_a_first_name_one_letter_off_takes_the_roster_spelling():
    assert spell_names("Happy birthday, Nima. Sam and Kevin came.", ["Nina Shah", "Sam Torres"]) == (
        "Happy birthday, Nina. Sam and Kevin came."
    )


def test_first_names_that_start_alike_match_under_one_last_name():
    assert spell_names("My coworker Kay Russo runs clinics.", ["Kate Russo"]) == "My coworker Kate Russo runs clinics."
    assert spell_names("Kim Russo came.", ["Kate Russo"]) == "Kim Russo came."
