import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "strategy_opt_20260721"


def test_post_v17_research_configs_contain_no_gap_or_current_open_size_factor():
    paths = []
    for version in tuple(f"v{number}" for number in range(18, 27)):
        paths.extend(RESULTS.glob(f"{version}_*_config.json"))
    assert paths

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        weights = payload["individual_config"]["weights"]
        assert all("Gap" not in name for name in weights), path
        assert "OvernightGapDown" not in weights, path
        assert "TrueMarketCap" not in weights, path
        assert (
            "TrueMarketCapT1" in weights or "PreCloseMarketCap" in weights
        ), path


def test_rejected_v17_has_no_deployable_config():
    assert not (ROOT / "configs" / "smallcap_v17_overnight_gap_w20_config.json").exists()
