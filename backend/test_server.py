"""
Offline checks for the director's deterministic paths.
Run: python backend/test_server.py

The model is not exercised here — it needs Ollama and it is allowed to be
fuzzy. What must never be fuzzy is the fallback that catches it: if the model
is down, garbled, or wrong, these rules are the whole product.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from server import _coerce, _fallback


def test_explicit_coordinates_are_parsed():
    out = _fallback("51.5, -0.12")
    assert out["intent"] == "locate", out
    assert out["params"] == {"lat": 51.5, "lon": -0.12}, out


def test_out_of_range_coordinates_are_not_treated_as_a_place():
    # 99 is not a latitude. Better to do nothing than to mark a wrong dot.
    assert _fallback("99, 200")["intent"] == "idle"


def test_locate_phrasings():
    for text in ["show my location", "where am I", "find me", "locate me",
                 "WHERE AM I?", "can you show me where i am"]:
        assert _fallback(text)["intent"] == "locate", text


def test_unrelated_text_is_idle():
    for text in ["hello", "", "asdfgh", "what is the capital of France"]:
        assert _fallback(text)["intent"] == "idle", text


def test_coerce_rejects_anything_off_schema():
    assert _coerce('{"intent":"locate"}')["intent"] == "locate"
    assert _coerce('{"intent":"locate","params":{"lat":1,"lon":2}}')["params"]["lat"] == 1
    # An intent the frontend has no branch for must not reach it.
    assert _coerce('{"intent":"launch_missiles"}') is None
    assert _coerce("not json at all") is None
    assert _coerce("[1,2,3]") is None
    # Malformed params degrade to empty rather than propagating a bad type.
    assert _coerce('{"intent":"idle","params":"nope"}')["params"] == {}


def test_model_coordinates_are_stripped():
    # The model invents plausible coordinates. When the user typed none, they
    # must not survive into an intent the map will act on.
    raw = chr(39)
    out = _coerce("{\"intent\":\"locate\",\"params\":{\"lat\":37.77,\"lon\":-122.41}}",
                  allow_coords=False)
    assert out["params"] == {}, out
    kept = _coerce("{\"intent\":\"locate\",\"params\":{\"lat\":37.77,\"lon\":-122.41}}",
                   allow_coords=True)
    assert kept["params"]["lat"] == 37.77


if __name__ == "__main__":
    test_explicit_coordinates_are_parsed()
    test_out_of_range_coordinates_are_not_treated_as_a_place()
    test_locate_phrasings()
    test_unrelated_text_is_idle()
    test_coerce_rejects_anything_off_schema()
    test_model_coordinates_are_stripped()
    print("director fallback checks passed")
