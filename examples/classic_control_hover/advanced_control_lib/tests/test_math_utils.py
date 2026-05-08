"""Tests for math_utils: clamp."""

from advanced_control_lib.math_utils import clamp


class TestClamp:
    def test_within_bounds(self) -> None:
        assert clamp(3.0, 0.0, 5.0) == 3.0

    def test_below_lower(self) -> None:
        assert clamp(-1.0, 0.0, 5.0) == 0.0

    def test_above_upper(self) -> None:
        assert clamp(10.0, 0.0, 5.0) == 5.0

    def test_lower_none(self) -> None:
        assert clamp(-100.0, None, 5.0) == -100.0

    def test_upper_none(self) -> None:
        assert clamp(100.0, 0.0, None) == 100.0

    def test_both_none(self) -> None:
        assert clamp(42.0, None, None) == 42.0

    def test_at_lower_bound(self) -> None:
        assert clamp(0.0, 0.0, 5.0) == 0.0

    def test_at_upper_bound(self) -> None:
        assert clamp(5.0, 0.0, 5.0) == 5.0

    def test_negative_bounds(self) -> None:
        assert clamp(0.0, -5.0, -1.0) == -1.0

    def test_crossed_bounds_clamp_to_lower(self) -> None:
        """With lower > upper, value is clamped to lower (first bound applied)."""
        assert clamp(2.0, 5.0, 0.0) == 0.0  # min(max(2,5), 0) = min(5,0) = 0
