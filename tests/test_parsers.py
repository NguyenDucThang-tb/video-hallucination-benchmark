from src.evaluation.parsers import parse_ab_ba, parse_mcq, parse_yes_no


def test_yes_no_parser():
    assert parse_yes_no("Yes.").value == "yes"
    assert parse_yes_no("No, that is absent.").value == "no"
    assert parse_yes_no("Yes and no").status == "ambiguous"
    assert parse_yes_no("Maybe").status == "unparseable"


def test_mcq_parser():
    assert parse_mcq("The answer is C.").value == "C"
    assert parse_mcq("red car", {"A": "blue car", "B": "red car"}).value == "B"
    assert parse_mcq("A or B").status == "ambiguous"


def test_tsh_parser():
    assert parse_ab_ba("AB").value == "AB"
    assert parse_ab_ba("Answer: BA").value == "BA"
    assert parse_ab_ba("Action A happens before Action B.").value == "AB"
    assert parse_ab_ba("Action A happens after Action B.").value == "BA"
    assert parse_ab_ba("A then B").status == "unparseable"
