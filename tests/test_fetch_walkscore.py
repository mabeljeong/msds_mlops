"""
Integration tests (mocked HTTP) for scripts/fetch_walkscore.py.
"""

import os

os.environ.setdefault("WALKSCORE_API_KEY", "test-walkscore-key")

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import requests

from scripts import fetch_walkscore
from scripts.fetch_walkscore import (
    SCORE_COLUMNS,
    SF_CITY_CENTROID,
    WS_STATUS_INTERNAL_ERROR,
    WS_STATUS_INVALID_KEY,
    WS_STATUS_INVALID_LATLON,
    WS_STATUS_IN_PROGRESS,
    WS_STATUS_IP_BLOCKED,
    WS_STATUS_QUOTA_EXCEEDED,
    WalkScoreAPIError,
    _NULL_SCORES,
    describe_score,
    enrich_walkscore,
    geocode_address,
    geocode_with_fallback,
    get_scores,
    get_scores_with_retry,
    impute_missing_scores,
    recover_address,
)


SAMPLE_PAYLOAD = {
    "status": 1,
    "walkscore": 88,
    "description": "Very Walkable",
    "transit": {"score": 70, "description": "Excellent Transit"},
    "bike": {"score": 65, "description": "Bikeable"},
}


@pytest.fixture(autouse=True)
def _clear_geocode_cache():
    """Reset module-level caches between tests for deterministic behavior."""
    fetch_walkscore._geocode_cache.clear()
    fetch_walkscore._score_cache.clear()
    yield
    fetch_walkscore._geocode_cache.clear()
    fetch_walkscore._score_cache.clear()


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
        assert result is not _NULL_SCORES

    def test_failure_fallback_on_http_error(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500")

        with patch.object(fetch_walkscore.requests, "get", return_value=mock_resp):
            result = get_scores("39 Bruton St", "San Francisco", 37.78, -122.42)

        assert result == _NULL_SCORES

    @pytest.mark.parametrize(
        "transient_status",
        [WS_STATUS_IN_PROGRESS, WS_STATUS_INTERNAL_ERROR, WS_STATUS_INVALID_LATLON],
    )
    def test_transient_or_invalid_coords_status_returns_nulls(self, transient_status):
        """Statuses 2 / 31 (transient) and 30 (bad coords) must surface as nulls
        so the retry layer / coord-fallback path can take over."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": transient_status}
        mock_resp.raise_for_status.return_value = None

        with patch.object(fetch_walkscore.requests, "get", return_value=mock_resp):
            result = get_scores("39 Bruton St", "San Francisco", 37.78, -122.42)

        assert result == _NULL_SCORES

    @pytest.mark.parametrize(
        "fatal_status",
        [WS_STATUS_INVALID_KEY, WS_STATUS_QUOTA_EXCEEDED, WS_STATUS_IP_BLOCKED],
    )
    def test_fatal_status_raises_walkscore_api_error(self, fatal_status):
        """Permanent API failures (40/41/42) must raise so the pipeline never
        silently writes blank score columns."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": fatal_status}
        mock_resp.raise_for_status.return_value = None

        with patch.object(fetch_walkscore.requests, "get", return_value=mock_resp):
            with pytest.raises(WalkScoreAPIError) as excinfo:
                get_scores("39 Bruton St", "San Francisco", 37.78, -122.42)

        assert excinfo.value.status == fatal_status


class TestGeocodeAddress:
    def test_returns_lat_lon_on_success(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"lat": "37.78", "lon": "-122.42"}]
        mock_resp.raise_for_status.return_value = None

        with patch.object(fetch_walkscore.requests, "get", return_value=mock_resp) as mock_get:
            lat, lon = geocode_address("123 Market St", "San Francisco", delay=0)

        assert (lat, lon) == (37.78, -122.42)
        args, kwargs = mock_get.call_args
        assert args[0] == fetch_walkscore.NOMINATIM_URL
        assert kwargs["params"]["q"] == "123 Market St, San Francisco"
        assert kwargs["headers"]["User-Agent"] == fetch_walkscore.GEOCODE_USER_AGENT

    def test_caches_repeated_lookups(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"lat": "1.0", "lon": "2.0"}]
        mock_resp.raise_for_status.return_value = None

        with patch.object(fetch_walkscore.requests, "get", return_value=mock_resp) as mock_get:
            a = geocode_address("123 Market St", "San Francisco", delay=0)
            b = geocode_address("123 Market St", "San Francisco", delay=0)

        assert a == b == (1.0, 2.0)
        assert mock_get.call_count == 1

    def test_empty_results_returns_none_pair(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None

        with patch.object(fetch_walkscore.requests, "get", return_value=mock_resp):
            assert geocode_address("nowhere", delay=0) == (None, None)

    def test_request_exception_returns_none_pair(self):
        with patch.object(
            fetch_walkscore.requests,
            "get",
            side_effect=requests.RequestException("boom"),
        ):
            assert geocode_address("123 Market St", delay=0) == (None, None)

    def test_missing_address_short_circuits(self):
        with patch.object(fetch_walkscore.requests, "get") as mock_get:
            assert geocode_address("", delay=0) == (None, None)
            assert geocode_address(None, delay=0) == (None, None)  # type: ignore[arg-type]
        mock_get.assert_not_called()


class TestRecoverAddress:
    def test_uses_address_when_addressable(self):
        addr, src = recover_address(
            address="123 Market St, San Francisco, CA 94103",
            url="https://x/y",
            zip_code="94103",
        )
        assert addr == "123 Market St, San Francisco, CA 94103"
        assert src == "address"

    def test_strips_unit_at_prefix(self):
        addr, src = recover_address(address="Unit 1315 at 388 Beale")
        assert addr == "388 Beale"
        assert src == "address_normalized"

    def test_falls_back_to_url_slug(self):
        url = "https://www.apartments.com/14-isis-st-san-francisco-ca-unit-4/3g0pk3f/"
        addr, src = recover_address(address="", url=url, zip_code=None)
        assert addr == "14 isis st"
        assert src == "url"

    def test_skips_pure_hash_url_segment(self):
        url = "https://www.apartments.com/yp192jv/"
        addr, src = recover_address(address=None, url=url, zip_code="94130")
        # No street info parseable; falls back to ZIP-centroid query.
        assert src == "zip_fallback"
        assert "94130" in addr

    def test_zip_fallback_normalizes_numeric_zip(self):
        addr, src = recover_address(address="", url="", zip_code=94110)
        assert src == "zip_fallback"
        assert "94110" in addr
        assert "San Francisco" in addr

    def test_returns_none_when_nothing_usable(self):
        addr, src = recover_address(address=None, url=None, zip_code=None)
        assert addr is None
        assert src == "none"

    def test_address_with_only_letters_falls_through(self):
        # No digits → not addressable → fallback to url/zip.
        addr, src = recover_address(
            address="Spera Apartments", url=None, zip_code="94158"
        )
        assert src == "zip_fallback"
        assert "94158" in addr


class TestGeocodeWithFallback:
    def test_uses_address_when_geocode_succeeds(self):
        with patch.object(
            fetch_walkscore, "geocode_address", return_value=(37.78, -122.42)
        ) as gc:
            lat, lon, src = geocode_with_fallback(
                "123 Market St", "San Francisco", "94103", delay=0
            )
        assert (lat, lon) == (37.78, -122.42)
        assert src == "address"
        assert gc.call_count == 1

    def test_falls_back_to_zip_centroid(self):
        # First call (street): None. Second call (zip): success.
        results = iter([(None, None), (37.76, -122.43)])

        def fake_geocode(*_args, **_kwargs):
            return next(results)

        with patch.object(fetch_walkscore, "geocode_address", side_effect=fake_geocode):
            lat, lon, src = geocode_with_fallback(
                "bogus", "San Francisco", "94110", delay=0
            )
        assert (lat, lon) == (37.76, -122.43)
        assert src == "zip"

    def test_falls_back_to_city_centroid_when_all_fail(self):
        with patch.object(
            fetch_walkscore, "geocode_address", return_value=(None, None)
        ):
            lat, lon, src = geocode_with_fallback(
                "bogus", "San Francisco", "00000", delay=0
            )
        assert (lat, lon) == SF_CITY_CENTROID
        assert src == "city"

    def test_no_address_skips_to_zip(self):
        with patch.object(
            fetch_walkscore, "geocode_address", return_value=(37.7, -122.4)
        ) as gc:
            lat, lon, src = geocode_with_fallback(
                None, "San Francisco", "94110", delay=0
            )
        assert src == "zip"
        # geocode_address called exactly once (zip path), not for missing address.
        assert gc.call_count == 1


class TestGetScoresWithRetry:
    def test_returns_immediately_on_success(self):
        with patch.object(
            fetch_walkscore,
            "get_scores",
            return_value={
                "walk_score": 90,
                "walk_description": "Walker's Paradise",
                "transit_score": 80,
                "transit_description": "Excellent Transit",
                "bike_score": 70,
                "bike_description": "Very Bikeable",
            },
        ) as gs, patch.object(fetch_walkscore.time, "sleep"):
            result = get_scores_with_retry("a", "c", 1.0, 2.0, max_retries=2, backoff=0)
        assert gs.call_count == 1
        assert result["walk_score"] == 90

    def test_retries_when_walk_score_null(self):
        sequence = [
            dict(_NULL_SCORES),
            dict(_NULL_SCORES),
            {
                "walk_score": 50,
                "walk_description": "Somewhat Walkable",
                "transit_score": 40,
                "transit_description": "Some Transit",
                "bike_score": 30,
                "bike_description": "Somewhat Bikeable",
            },
        ]
        with patch.object(
            fetch_walkscore, "get_scores", side_effect=sequence
        ) as gs, patch.object(fetch_walkscore.time, "sleep"):
            result = get_scores_with_retry("a", "c", 1.0, 2.0, max_retries=2, backoff=0)
        assert gs.call_count == 3
        assert result["walk_score"] == 50

    def test_returns_nulls_after_exhausting_retries(self):
        with patch.object(
            fetch_walkscore, "get_scores", return_value=dict(_NULL_SCORES)
        ) as gs, patch.object(fetch_walkscore.time, "sleep"):
            result = get_scores_with_retry("a", "c", 1.0, 2.0, max_retries=2, backoff=0)
        assert gs.call_count == 3
        assert result == _NULL_SCORES

    def test_walkscore_api_error_propagates_without_retries(self):
        """Permanent API errors should not be retried — they cannot be fixed
        by reissuing the same request."""
        with patch.object(
            fetch_walkscore,
            "get_scores",
            side_effect=WalkScoreAPIError(WS_STATUS_INVALID_KEY, "bad key"),
        ) as gs, patch.object(fetch_walkscore.time, "sleep"):
            with pytest.raises(WalkScoreAPIError):
                get_scores_with_retry("a", "c", 1.0, 2.0, max_retries=2, backoff=0)
        assert gs.call_count == 1

    def test_retries_on_transient_api_status_then_succeeds(self):
        """Walk Score JSON status 2 (in_progress) returns null scores from
        get_scores; the retry wrapper should reissue and eventually succeed."""

        def make_resp(payload):
            r = MagicMock()
            r.json.return_value = payload
            r.raise_for_status.return_value = None
            return r

        responses = [
            make_resp({"status": WS_STATUS_IN_PROGRESS}),
            make_resp({"status": WS_STATUS_IN_PROGRESS}),
            make_resp(SAMPLE_PAYLOAD),
        ]
        with patch.object(
            fetch_walkscore.requests, "get", side_effect=responses
        ), patch.object(fetch_walkscore.time, "sleep"):
            result = get_scores_with_retry(
                "39 Bruton St", "San Francisco", 37.78, -122.42,
                max_retries=2, backoff=0,
            )
        assert result["walk_score"] == 88


class TestDescribeScore:
    @pytest.mark.parametrize(
        "score,kind,expected",
        [
            (95, "walk", "Walker's Paradise"),
            (75, "walk", "Very Walkable"),
            (55, "walk", "Somewhat Walkable"),
            (30, "walk", "Car-Dependent"),
            (5, "walk", "Car-Dependent"),
            (95, "transit", "Rider's Paradise"),
            (75, "transit", "Excellent Transit"),
            (55, "transit", "Good Transit"),
            (30, "transit", "Some Transit"),
            (5, "transit", "Minimal Transit"),
            (95, "bike", "Biker's Paradise"),
            (75, "bike", "Very Bikeable"),
            (55, "bike", "Bikeable"),
        ],
    )
    def test_buckets(self, score, kind, expected):
        assert describe_score(score, kind) == expected

    def test_unknown_kind_returns_none(self):
        assert describe_score(50, "swim") is None

    def test_missing_score_returns_none(self):
        assert describe_score(None, "walk") is None
        assert describe_score(np.nan, "walk") is None


class TestImputeMissingScores:
    def test_zip_group_then_global_fill(self):
        df = pd.DataFrame(
            {
                "zip_code": ["94110", "94110", "94103", "94103"],
                "walk_score": [80, None, 60, 50],
                "walk_description": ["Very Walkable", None, "Somewhat Walkable", None],
                "transit_score": [None, None, 70, 50],
                "transit_description": [None, None, "Excellent Transit", None],
                "bike_score": [None, 60, None, None],
                "bike_description": [None, "Bikeable", None, None],
            }
        )
        out = impute_missing_scores(df.copy())

        for col in (
            "walk_score",
            "walk_description",
            "transit_score",
            "transit_description",
            "bike_score",
            "bike_description",
        ):
            assert out[col].notna().all(), f"{col} still has nulls"

        # 94110 row 1 walk_score: zip mean of 94110 = 80
        assert out.loc[1, "walk_score"] == 80
        # 94110 transit_score: no values in zip; falls back to global mean of (70+50)/2 = 60
        assert out.loc[0, "transit_score"] == 60
        # bike_description for filled scores derived from buckets
        assert out.loc[1, "bike_description"] == "Bikeable"

    def test_descriptions_only_fills_when_score_present(self):
        df = pd.DataFrame(
            {
                "walk_score": [85, None],
                "walk_description": [None, None],
                "transit_score": [None, None],
                "transit_description": [None, None],
                "bike_score": [None, None],
                "bike_description": [None, None],
            }
        )
        out = impute_missing_scores(df.copy())
        # When everything is None and no zip column, description still fills
        # only after score gets imputed (global mean of 85 = 85 fills row 1).
        assert out.loc[0, "walk_description"] == "Very Walkable"
        assert out.loc[1, "walk_score"] == 85
        assert out.loc[1, "walk_description"] == "Very Walkable"


class TestEnrichWalkscore:
    def test_geocodes_when_coords_missing_and_appends_scores(self):
        df = pd.DataFrame(
            {
                "address": ["123 Market St", "500 Valencia St"],
                "url": ["u1", "u2"],
            }
        )

        def fake_geocode(address, city, **_kwargs):
            mapping = {
                "123 Market St, San Francisco": (37.78, -122.41),
                "500 Valencia St, San Francisco": (37.76, -122.42),
            }
            return mapping.get(f"{address}, {city}", (37.78, -122.41))

        def fake_get_scores(address, city, lat, lon):
            return {
                "walk_score": 90,
                "walk_description": "Walker's Paradise",
                "transit_score": 80,
                "transit_description": "Excellent Transit",
                "bike_score": 75,
                "bike_description": "Very Bikeable",
            }

        with patch.object(fetch_walkscore, "geocode_address", side_effect=fake_geocode), \
             patch.object(fetch_walkscore, "get_scores", side_effect=fake_get_scores) as ws:
            out = enrich_walkscore(df, delay=0, geocode_delay=0)

        assert list(out.columns)[-len(SCORE_COLUMNS):] == list(SCORE_COLUMNS)
        assert ws.call_count == 2
        assert (out["walk_score"] == 90).all()
        assert (out["transit_score"] == 80).all()
        assert (out["bike_description"] == "Very Bikeable").all()

    def test_skips_geocode_when_coords_present(self):
        df = pd.DataFrame(
            {
                "address": ["123 Market St"],
                "latitude": [37.78],
                "longitude": [-122.42],
            }
        )

        with patch.object(fetch_walkscore, "geocode_address") as gc, \
             patch.object(
                 fetch_walkscore,
                 "get_scores",
                 return_value={
                     "walk_score": 88,
                     "walk_description": "Very Walkable",
                     "transit_score": 70,
                     "transit_description": "Excellent Transit",
                     "bike_score": 65,
                     "bike_description": "Bikeable",
                 },
             ):
            out = enrich_walkscore(df, delay=0, geocode_delay=0)

        gc.assert_not_called()
        assert out.loc[0, "walk_score"] == 88

    def test_failed_geocode_falls_back_to_city_centroid(self):
        """New behavior: failures are recovered via city centroid; scores are still attempted."""
        df = pd.DataFrame({"address": ["nowhere"]})

        with patch.object(fetch_walkscore, "geocode_address", return_value=(None, None)), \
             patch.object(
                 fetch_walkscore,
                 "get_scores",
                 return_value={
                     "walk_score": 70,
                     "walk_description": "Very Walkable",
                     "transit_score": 60,
                     "transit_description": "Good Transit",
                     "bike_score": 55,
                     "bike_description": "Bikeable",
                 },
             ) as ws:
            out = enrich_walkscore(df, delay=0, geocode_delay=0)

        # get_scores IS called (centroid fallback supplies coords).
        assert ws.call_count == 1
        _args, _kwargs = ws.call_args
        # Lat/lon are the city centroid.
        assert _args[2] == SF_CITY_CENTROID[0]
        assert _args[3] == SF_CITY_CENTROID[1]
        assert out.loc[0, "walk_score"] == 70

    def test_ensure_complete_imputes_missing_fields(self):
        """When the API only returns walk_score, ensure_complete fills the rest."""
        df = pd.DataFrame(
            {
                "address": ["123 Market St", "500 Valencia St"],
                "zip_code": ["94103", "94110"],
            }
        )

        responses = [
            {  # row 0 — full payload
                "walk_score": 95,
                "walk_description": "Walker's Paradise",
                "transit_score": 85,
                "transit_description": "Excellent Transit",
                "bike_score": 75,
                "bike_description": "Very Bikeable",
            },
            {  # row 1 — only walk_score
                "walk_score": 60,
                "walk_description": None,
                "transit_score": None,
                "transit_description": None,
                "bike_score": None,
                "bike_description": None,
            },
        ]

        with patch.object(
            fetch_walkscore, "geocode_address", return_value=(37.78, -122.42)
        ), patch.object(
            fetch_walkscore, "get_scores", side_effect=responses
        ):
            out = enrich_walkscore(df, delay=0, geocode_delay=0, ensure_complete=True)

        for col in SCORE_COLUMNS:
            assert out[col].notna().all(), f"{col} still has nulls"
        # Row 1 walk_description bucket-derived from score 60.
        assert out.loc[1, "walk_description"] == "Somewhat Walkable"

    def test_no_address_url_or_zip_columns_short_circuits(self):
        df = pd.DataFrame({"rent_usd": [3000]})
        out = enrich_walkscore(df, delay=0, geocode_delay=0)
        for col in SCORE_COLUMNS:
            assert col in out.columns
            assert pd.isna(out.loc[0, col])

    def test_falls_back_to_city_centroid_when_first_score_call_fails(self):
        """When the first get_scores call returns null but the row used non-
        centroid coordinates, the enricher should retry once at the SF city
        centroid before relying on imputation."""
        df = pd.DataFrame(
            {
                "address": ["123 Market St"],
                "latitude": [37.78],
                "longitude": [-122.42],
            }
        )

        responses = [
            dict(_NULL_SCORES),
            {
                "walk_score": 80,
                "walk_description": "Very Walkable",
                "transit_score": 70,
                "transit_description": "Excellent Transit",
                "bike_score": 60,
                "bike_description": "Bikeable",
            },
        ]
        captured_calls = []

        def fake_get_scores(address, city, lat, lon):
            captured_calls.append((address, city, lat, lon))
            return responses[len(captured_calls) - 1]

        with patch.object(
            fetch_walkscore, "get_scores", side_effect=fake_get_scores
        ), patch.object(fetch_walkscore.time, "sleep"):
            out = enrich_walkscore(df, delay=0, geocode_delay=0, score_retries=0)

        # Two distinct score requests: original coords, then SF centroid.
        assert len(captured_calls) == 2
        assert (round(captured_calls[1][2], 6), round(captured_calls[1][3], 6)) == (
            round(SF_CITY_CENTROID[0], 6),
            round(SF_CITY_CENTROID[1], 6),
        )
        assert out.loc[0, "walk_score"] == 80

    def test_ensure_complete_fills_walk_score_when_api_returns_nothing(self):
        """``ensure_complete`` must guarantee a walk_score for every row even
        when both the primary and fallback score calls fail."""
        df = pd.DataFrame(
            {
                "address": ["123 Market St", "500 Valencia St"],
                "zip_code": ["94103", "94110"],
            }
        )

        responses = [
            {  # row 0 first attempt
                "walk_score": 90,
                "walk_description": "Walker's Paradise",
                "transit_score": 80,
                "transit_description": "Excellent Transit",
                "bike_score": 70,
                "bike_description": "Very Bikeable",
            },
            dict(_NULL_SCORES),  # row 1 first attempt
            dict(_NULL_SCORES),  # row 1 fallback (city centroid)
        ]

        with patch.object(
            fetch_walkscore, "geocode_address", return_value=(37.78, -122.42)
        ), patch.object(
            fetch_walkscore, "get_scores", side_effect=responses
        ), patch.object(fetch_walkscore.time, "sleep"):
            out = enrich_walkscore(
                df, delay=0, geocode_delay=0, ensure_complete=True, score_retries=0,
            )

        # walk_score must be non-null for both rows; row 1 imputes from
        # row 0's value (only data point) via global mean fallback.
        assert out["walk_score"].notna().all()
        assert out.loc[0, "walk_score"] == 90
        assert out.loc[1, "walk_score"] == 90
