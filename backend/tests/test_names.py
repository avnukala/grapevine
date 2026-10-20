from grapevine.names import spell_names

ROSTER = ["Sam Torres", "Sam Reyes", "Jasper Lin", "Kate Russo"]


def test_near_misses_of_roster_names_take_the_roster_spelling():
    assert spell_names("San Torres, and that is Jasper Lind.", ROSTER) == "Sam Torres, and that is Jasper Lin."


def test_other_names_and_far_misses_stay_as_heard():
    text = "Kay Russo met Sam Reyes and Andrew Gao at Red Rocks."
    assert spell_names(text, ROSTER) == text
