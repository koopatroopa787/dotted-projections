"""
Offline checks for the director's routing.
Run: python backend/test_server.py

Every input has to land on an intent — typing "cat" and watching nothing happen
is the failure mode this file exists to prevent. The model is not exercised
here: it only writes chat replies now, and those are allowed to be fuzzy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from server import _fallback  # noqa: E402


def test_explicit_coordinates_are_parsed():
    out = _fallback("51.5, -0.12")
    assert out["intent"] == "locate", out
    assert out["params"] == {"lat": 51.5, "lon": -0.12}, out


def test_out_of_range_coordinates_are_not_treated_as_a_place():
    # 99 is not a latitude, so this is not a location — it becomes a shape.
    assert _fallback("99, 200")["intent"] != "locate"


def test_locate_phrasings():
    for text in ["show my location", "where am I", "find me", "locate me",
                 "WHERE AM I?", "can you show me where i am"]:
        assert _fallback(text)["intent"] == "locate", text


def test_shape_requests_carry_the_word():
    for text, word in [("draw a dog", "dog"), ("draw dog", "dog"),
                       ("sketch a heart", "heart"), ("spell hello", "hello"),
                       ("make an octopus", "octopus"),
                       ("show me a rocket", "rocket")]:
        out = _fallback(text)
        assert out["intent"] == "shape", (text, out)
        assert out["params"]["word"] == word, (text, out)


def test_show_me_a_thing_is_not_a_locate():
    # "show me a dog" contains "show me", which the locate pattern also matches.
    assert _fallback("show me a dog")["intent"] == "shape"
    assert _fallback("show me my location")["intent"] == "locate"
    assert _fallback("draw my location")["intent"] == "locate"


def test_every_bare_word_draws_something():
    # The whole point: no input is a no-op.
    for text in ["cat", "wow", "banana", "asdfgh", "hello", "42", "moon rise"]:
        out = _fallback(text)
        assert out["intent"] == "shape", (text, out)
        assert out["params"]["word"], (text, out)


def test_reverse_is_recognised():
    for text in ["reverse", "invert the map", "go negative"]:
        assert _fallback(text)["intent"] == "reverse", text


def test_prose_and_questions_become_chat():
    for text in ["talk to me", "what do you think of all this",
                 "how are you today?", "say something"]:
        assert _fallback(text)["intent"] == "chat", text


if __name__ == "__main__":
    test_explicit_coordinates_are_parsed()
    test_out_of_range_coordinates_are_not_treated_as_a_place()
    test_locate_phrasings()
    test_shape_requests_carry_the_word()
    test_show_me_a_thing_is_not_a_locate()
    test_every_bare_word_draws_something()
    test_reverse_is_recognised()
    test_prose_and_questions_become_chat()
    print("director routing checks passed")
