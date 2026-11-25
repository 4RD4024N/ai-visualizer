import pytest

from lyrics import clean_line, overlap


@pytest.mark.parametrize("text", [
    "Thank you for watching!", "Altyazı M.K.", "Subtitles by the Amara.org community",
    "la la la la la la", "Oh yeah yeah", "♪ ♪ ♪", "Music", "you", "",
])
def test_hallucinations_and_fillers_are_rejected(text):
    assert clean_line(text) is None


@pytest.mark.parametrize("text", [
    "I'm walking on the burning streets tonight",
    "Your eyes are like the ocean in the night",
    "We will dance until the sun comes up",
])
def test_real_lines_pass(text):
    assert clean_line(text) == text


def test_overlapping_windows_are_same_line():
    assert overlap("your eyes are like the ocean", "your eyes are like the ocean in the night") >= 0.6
    assert overlap("your eyes are like the ocean", "we will dance until the sun comes up") < 0.6
