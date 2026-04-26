"""
Unit tests for step1_parse_kvn.py — specifically the parse_kvn_file() function.

Uses minimal synthetic KVN content so no real data files are needed.

Run with: pytest Scripts/tests/test_parser.py -v
"""

import sys
import os
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from step1_parse_kvn import parse_kvn_file


# ---------------------------------------------------------------------------
# Minimal valid KVN fixture
# ---------------------------------------------------------------------------

MINIMAL_KVN = """CCSDS_CDM_VERS = 1.0
CREATION_DATE = 2025-11-01T12:00:00.000
ORIGINATOR = KAU_TEAM
MESSAGE_ID = CDM_12345_67890_001
TCA = 2025-11-03T10:00:00.000
MISS_DISTANCE = 150.5 [m]
RELATIVE_SPEED = 12500.0 [m/s]
COLLISION_PROBABILITY = 1.5E-05
COLLISION_PROBABILITY_METHOD = FOSTER-1992
OBJECT = OBJECT1
OBJECT_DESIGNATOR = 12345
OBJECT_NAME = SAT-A
OBJECT_TYPE = PAYLOAD
CR_R = 25.0 [m**2]
CT_T = 100.0 [m**2]
CN_N = 50.0 [m**2]
X = 6500.0 [km]
Y = 1200.0 [km]
Z = 500.0 [km]
X_DOT = 1.5 [km/s]
Y_DOT = -7.2 [km/s]
Z_DOT = 0.3 [km/s]
OBJECT = OBJECT2
OBJECT_DESIGNATOR = 67890
OBJECT_NAME = DEB-B
OBJECT_TYPE = DEBRIS
CR_R = 10.0 [m**2]
CT_T = 40.0 [m**2]
CN_N = 20.0 [m**2]
X = 6499.9 [km]
Y = 1200.1 [km]
Z = 500.1 [km]
X_DOT = -1.3 [km/s]
Y_DOT = 7.1 [km/s]
Z_DOT = -0.2 [km/s]
"""


def write_temp_kvn(content, filename='CDM_12345_67890_001.kvn'):
    """Write KVN content to a temp file and return the path."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParseKvnFile:

    def test_returns_dict_on_valid_file(self):
        """Valid KVN file should return a non-None dict."""
        path = write_temp_kvn(MINIMAL_KVN)
        result = parse_kvn_file(path)
        assert result is not None
        assert isinstance(result, dict)

    def test_extracts_collision_probability(self):
        """COLLISION_PROBABILITY should be extracted correctly."""
        path = write_temp_kvn(MINIMAL_KVN)
        result = parse_kvn_file(path)
        assert 'COLLISION_PROBABILITY' in result
        assert float(result['COLLISION_PROBABILITY']) == pytest.approx(1.5e-5, rel=1e-3)

    def test_extracts_miss_distance(self):
        """MISS_DISTANCE should be extracted and units stripped."""
        path = write_temp_kvn(MINIMAL_KVN)
        result = parse_kvn_file(path)
        assert 'MISS_DISTANCE' in result
        assert float(result['MISS_DISTANCE']) == pytest.approx(150.5, rel=1e-3)

    def test_extracts_tca(self):
        """TCA should be extracted as a string."""
        path = write_temp_kvn(MINIMAL_KVN)
        result = parse_kvn_file(path)
        assert 'TCA' in result
        assert '2025-11-03' in result['TCA']

    def test_object1_covariance_prefixed(self):
        """Object1 covariance should be prefixed with 'object1_'."""
        path = write_temp_kvn(MINIMAL_KVN)
        result = parse_kvn_file(path)
        assert 'object1_CR_R' in result
        assert float(result['object1_CR_R']) == pytest.approx(25.0, rel=1e-3)

    def test_object2_covariance_prefixed(self):
        """Object2 covariance should be prefixed with 'object2_'."""
        path = write_temp_kvn(MINIMAL_KVN)
        result = parse_kvn_file(path)
        assert 'object2_CR_R' in result
        assert float(result['object2_CR_R']) == pytest.approx(10.0, rel=1e-3)

    def test_units_stripped_from_values(self):
        """Values with units like [m] should have units removed."""
        path = write_temp_kvn(MINIMAL_KVN)
        result = parse_kvn_file(path)
        # Should not contain '[m]' or '[m**2]' in any value
        for key, val in result.items():
            if isinstance(val, str):
                assert '[' not in val, f"Unit not stripped from {key}: {val}"

    def test_event_id_extracted_from_filename(self):
        """event_id should be 'obj1_obj2' extracted from CDM_obj1_obj2_*.kvn filename."""
        path = write_temp_kvn(MINIMAL_KVN, filename='CDM_12345_67890_001.kvn')
        result = parse_kvn_file(path)
        assert result['event_id'] == '12345_67890'

    def test_object1_norad_from_filename(self):
        """object1_norad_id should be extracted from filename."""
        path = write_temp_kvn(MINIMAL_KVN, filename='CDM_12345_67890_001.kvn')
        result = parse_kvn_file(path)
        assert result['object1_norad_id'] == '12345'

    def test_object2_norad_from_filename(self):
        """object2_norad_id should be extracted from filename."""
        path = write_temp_kvn(MINIMAL_KVN, filename='CDM_12345_67890_001.kvn')
        result = parse_kvn_file(path)
        assert result['object2_norad_id'] == '67890'

    def test_returns_none_on_nonexistent_file(self):
        """Non-existent file should return None (not crash)."""
        result = parse_kvn_file('/nonexistent/path/file.kvn')
        assert result is None

    def test_returns_none_on_empty_file(self):
        """Empty file should return a dict (no crash), may have no fields."""
        path = write_temp_kvn('', filename='CDM_11111_22222_001.kvn')
        result = parse_kvn_file(path)
        # Should not raise — either None or empty-ish dict
        assert result is None or isinstance(result, dict)

    def test_comment_lines_ignored(self):
        """Lines starting with COMMENT should be skipped."""
        kvn_with_comments = "COMMENT This is a comment\n" + MINIMAL_KVN
        path = write_temp_kvn(kvn_with_comments)
        result = parse_kvn_file(path)
        assert result is not None
        assert 'COMMENT This is a comment' not in result
