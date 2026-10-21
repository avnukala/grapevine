from grapevine.names import better_spelling, spell_names

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


def test_last_names_that_start_alike_match_under_one_first_name():
    assert spell_names("So Sam Rae and I dated.", ["Sam Reyes"]) == "So Sam Reyes and I dated."
    assert spell_names("Sam Torres came.", ["Sam Reyes"]) == "Sam Torres came."


def test_the_letter_x_after_an_article_is_an_ex():
    assert spell_names("I'm the X. Malcolm X spoke.", []) == "I'm the ex. Malcolm X spoke."


def test_the_better_spelling_is_a_name_said_alone_else_the_longer_one():
    said = "Hannahness Kim and I met in 2017. Me, Sam, Hannah, and Kevin. Hannah Kim is great."
    assert better_spelling("Hannahness Kim", "Hannah Kim", said) == "Hannah Kim"
    said = "My coworker Kay Russo. Kate Russo, not Kate Miller. Does Megan know Kate is coming?"
    assert better_spelling("Kay Russo", "Kate Russo", said) == "Kate Russo"
    said = "The Sam Reyes story. So Sam Rae and I dated. Sam Rae was great."
    assert better_spelling("Sam Rae", "Sam Reyes", said) == "Sam Reyes"
