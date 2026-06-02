"""Adapter tests: calibration, sinks, CSV lazy imports, CLI."""

from __future__ import annotations

import json

import numpy as np
import pytest

from wayfault import estimate_wwr
from wayfault.adapters.outbound.calibrator_regression import RegressionCalibrator
from wayfault.adapters.outbound.credit_flat import FlatHazardCreditCurveSource
from wayfault.adapters.outbound.dependence_hullwhite import HullWhiteHazardModel
from wayfault.adapters.outbound.exposure_inmemory import InMemoryExposureSource
from wayfault.adapters.outbound.sinks import DictResultSink, JsonReportWriter
from wayfault.domain.errors import MissingDependencyError


def test_regression_calibrator_roundtrip() -> None:
    rng = np.random.default_rng(7)
    true_b, true_a = 0.6, np.log(0.02)
    v = rng.normal(size=5000)
    hazard = np.exp(true_a + true_b * v)
    params = RegressionCalibrator().fit(v, hazard)
    assert params["b"] == pytest.approx(true_b, abs=1e-6)
    assert params["a"] == pytest.approx(true_a, abs=1e-6)
    # round-trip into a dependence model
    model = HullWhiteHazardModel(b=params["b"])
    assert model.dependence_param() == pytest.approx(true_b, abs=1e-6)


def test_regression_calibrator_guards() -> None:
    from wayfault.domain.errors import ValidationError
    with pytest.raises(ValidationError):
        RegressionCalibrator().fit(np.array([1.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValidationError):
        RegressionCalibrator().fit(np.array([1.0, 2.0]), np.array([-1.0, 2.0]))


def test_sinks(tmp_path) -> None:
    tenors = [0.5, 1.0]
    src = InMemoryExposureSource(np.array([[1.0, 2.0], [1.0, 2.0]]), tenors)
    credit = FlatHazardCreditCurveSource(hazard=0.02, recovery=0.4)
    dict_sink = DictResultSink()
    estimate_wwr(src, credit, HullWhiteHazardModel(b=0.2), sink=dict_sink)
    assert dict_sink.result is not None
    assert "wwr_cva" in dict_sink.result

    path = tmp_path / "out.json"
    estimate_wwr(src, credit, HullWhiteHazardModel(b=0.2), sink=JsonReportWriter(str(path)))
    loaded = json.loads(path.read_text())
    assert loaded["model"] == "HullWhiteHazardModel"


def test_csv_exposure_source(tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(np.array([[1.0, 2.0], [3.0, 4.0]]), columns=["0.5", "1.0"])
    p = tmp_path / "cube.csv"
    frame.to_csv(p, index=False)
    from wayfault.adapters.outbound.exposure_csv import CsvExposureSource
    cube = CsvExposureSource(str(p)).load()
    assert cube.n_scenarios == 2
    np.testing.assert_allclose(cube.grid.times, [0.5, 1.0])


def test_csv_credit_source(tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"knot": [1.0, 3.0], "hazard": [0.02, 0.05]})
    p = tmp_path / "curve.csv"
    frame.to_csv(p, index=False)
    from wayfault.adapters.outbound.credit_csv import CsvCreditCurveSource
    curve = CsvCreditCurveSource(str(p), recovery=0.4).load()
    assert curve.recovery.value == pytest.approx(0.4)


def test_optional_missing_dependency() -> None:
    from wayfault.adapters.outbound._optional import require
    with pytest.raises(MissingDependencyError):
        require("definitely_not_a_real_pkg_xyz", "io")


def test_cli_end_to_end(tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    cube = pd.DataFrame(
        np.random.default_rng(0).normal(size=(500, 2)) + 1.0, columns=["0.5", "1.0"]
    )
    curve = pd.DataFrame({"knot": [1.0, 3.0], "hazard": [0.02, 0.03]})
    cp = tmp_path / "cube.csv"
    crp = tmp_path / "curve.csv"
    outp = tmp_path / "res.json"
    cube.to_csv(cp, index=False)
    curve.to_csv(crp, index=False)
    from wayfault.adapters.inbound.cli import main
    rc = main([
        "estimate", "--exposure", str(cp), "--credit", str(crp),
        "--model", "hullwhite", "--b", "0.5", "--out", str(outp),
    ])
    assert rc == 0
    assert json.loads(outp.read_text())["model"] == "HullWhiteHazardModel"
