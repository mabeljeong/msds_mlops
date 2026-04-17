"""
Integration tests (mocked HTTP) for scripts/fetch_walkscore.py.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from scripts import fetch_walkscore
from scripts.fetch_walkscore import _NULL_SCORES, get_scores


SAMPLE_PAYLOAD = {
    "status": 1,
    "walkscore": 88,
    "description": "Very Walkable",
    "transit": {"score": 70, "description": "Excellent Transit"},
    "bike": {"score": 65, "description": "Bikeable"},
}


class TestWalkscoreApiBehavior:
    def test_successful_response_parsing(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_PAYLOAD
        mock_resp.raise_for_status.return_value = None

        with patch.object(fetch_walkscore.requests, "get", return_value=mock_resp):
            result = get_scores("39 Bruton St", "San Francisco", 37.78, -122.42)

        assert result == {
            "walk_score": 88,
            "walk_description": "Very Walkable",
            "transit_score": 70,
            "transit_description": "Excellent Transit",
            "bike_score": 65,
            "bike_description": "Bikeable",
        }

    def test_request_correctness(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_PAYLOAD
        mock_resp.raise_for_status.return_value = None

        with patch.object(
            fetch_walkscore.requests, "get", return_value=mock_resp
        ) as mock_get:
            get_scores("39 Bruton St", "San Francisco", 37.78, -122.42)

        assert mock_get.call_count == 1
        args, kwargs = mock_get.call_args

        assert args[0] == "https://api.walkscore.com/score"
        assert kwargs["timeout"] == 15

        params = kwargs["params"]
        assert params["format"] == "json"
        assert params["address"] == "39 Bruton St, San Francisco"
        assert params["lat"] == 37.78
        assert params["lon"] == -122.42
        assert params["transit"] == 1
        assert params["bike"] == 1
        assert params["wsapikey"] == fetch_walkscore.API_KEY
        assert params["wsapikey"]  # not empty

    def test_failure_fallback_on_request_exception(self):
        with patch.object(
            fetch_walkscore.requests,
            "get",
            side_effect=requests.RequestException("boom"),
        ):
            result = get_scores("39 Bruton St", "San Francisco", 37.78, -122.42)

        assert result == _NULL_SCORES
        assert all(v is None for v in result.values())
        assert result is not _NULL_SCORES  # defensive: returns a fresh dict

    def test_failure_fallback_on_http_error(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500")

        with patch.object(fetch_walkscore.requests, "get", return_value=mock_resp):
            result = get_scores("39 Bruton St", "San Francisco", 37.78, -122.42)

        assert result == _NULL_SCORES
