"""
test_session_detector.py — DST-Aware Session Detection Test Suite
=================================================================
Proves correct session classification with DST-aware boundaries
for London and New York market sessions.

V5-INSIGNIA Institutional Certification.
"""

import pytest
import datetime
from core.session_detector import SessionDetector


class TestDSTSessionBoundaries:
    """Verifies DST-aware session boundaries."""

    def test_london_summer_opens_at_7_utc(self):
        """During EU summer DST, London should open at 07:00 UTC."""
        # July 15 = definitely EU summer DST
        dt = datetime.datetime(2025, 7, 15, 7, 30, tzinfo=datetime.timezone.utc)
        session = SessionDetector.get_session(dt)
        assert session == "LONDON", f"Expected LONDON at 07:30 UTC summer, got {session}"

    def test_london_winter_opens_at_8_utc(self):
        """During EU winter, London should open at 08:00 UTC."""
        # January 15 = definitely EU winter
        dt = datetime.datetime(2025, 1, 15, 7, 30, tzinfo=datetime.timezone.utc)
        session = SessionDetector.get_session(dt)
        assert session == "TOKYO", f"Expected TOKYO at 07:30 UTC winter, got {session}"

    def test_london_winter_8am_is_london(self):
        """At 08:00 UTC winter, session should be LONDON."""
        dt = datetime.datetime(2025, 1, 15, 8, 0, tzinfo=datetime.timezone.utc)
        session = SessionDetector.get_session(dt)
        assert session == "LONDON", f"Expected LONDON at 08:00 UTC winter, got {session}"

    def test_ny_summer_opens_at_12_utc(self):
        """During US summer DST, NY should open at 12:00 UTC → overlap starts."""
        dt = datetime.datetime(2025, 7, 15, 12, 30, tzinfo=datetime.timezone.utc)
        session = SessionDetector.get_session(dt)
        assert session == "LONDON/NY", f"Expected LONDON/NY at 12:30 UTC summer, got {session}"

    def test_ny_winter_opens_at_13_utc(self):
        """During US winter, NY overlap starts at 13:00 UTC."""
        dt = datetime.datetime(2025, 1, 15, 13, 0, tzinfo=datetime.timezone.utc)
        session = SessionDetector.get_session(dt)
        assert session == "LONDON/NY", f"Expected LONDON/NY at 13:00 UTC winter, got {session}"

    def test_weekend_detection(self):
        """Saturday should show (CLOSED) suffix."""
        dt = datetime.datetime(2025, 7, 12, 10, 0, tzinfo=datetime.timezone.utc)  # Saturday
        session = SessionDetector.get_session(dt)
        assert "(CLOSED)" in session

    def test_rollover_window(self):
        """21:00-24:00 UTC should be ROLLOVER."""
        dt = datetime.datetime(2025, 7, 14, 22, 0, tzinfo=datetime.timezone.utc)  # Monday
        session = SessionDetector.get_session(dt)
        assert session == "ROLLOVER"


class TestDSTHelpers:
    """Verifies internal DST detection logic."""

    def test_eu_dst_active_in_july(self):
        dt = datetime.datetime(2025, 7, 15, 12, 0, tzinfo=datetime.timezone.utc)
        assert SessionDetector._is_dst_active(dt, "EU") is True

    def test_eu_dst_inactive_in_january(self):
        dt = datetime.datetime(2025, 1, 15, 12, 0, tzinfo=datetime.timezone.utc)
        assert SessionDetector._is_dst_active(dt, "EU") is False

    def test_us_dst_active_in_july(self):
        dt = datetime.datetime(2025, 7, 15, 12, 0, tzinfo=datetime.timezone.utc)
        assert SessionDetector._is_dst_active(dt, "US") is True

    def test_us_dst_inactive_in_january(self):
        dt = datetime.datetime(2025, 1, 15, 12, 0, tzinfo=datetime.timezone.utc)
        assert SessionDetector._is_dst_active(dt, "US") is False


class TestSessionFiltering:
    """Verifies is_session_active with DST-aware sessions."""

    def test_london_allowed_during_overlap(self):
        """LONDON/NY overlap should match 'LONDON' in allowed list."""
        dt = datetime.datetime(2025, 7, 15, 13, 0, tzinfo=datetime.timezone.utc)
        assert SessionDetector.is_session_active(dt, allowed_sessions=["LONDON"]) is True

    def test_no_filter_allows_all(self):
        """No allowed_sessions means all sessions are active."""
        dt = datetime.datetime(2025, 7, 15, 3, 0, tzinfo=datetime.timezone.utc)
        assert SessionDetector.is_session_active(dt) is True
