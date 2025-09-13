# test_assessment_analysis.py
import os
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from Assessment_analysis.main import AssessmentAnalysis


# ---- Dummy linear model for pickling ----
class DummyLinearModel:
    """
    Minimal sklearn-like regressor with .predict(X) that returns a deterministic
    value based on input to make assertions possible.
    prediction = 0.1*Hours_Studied + 0.01*Attendance + 0.2*Previous_Scores
                 + 0.05*Tutoring_Sessions + 0.03*Physical_Activity
    """
    def predict(self, X: pd.DataFrame):
        w = {
            "Hours_Studied": 0.1,
            "Attendance": 0.01,
            "Previous_Scores": 0.2,
            "Tutoring_Sessions": 0.05,
            "Physical_Activity": 0.03,
        }
        vals = (
            w["Hours_Studied"] * X["Hours_Studied"]
            + w["Attendance"] * X["Attendance"]
            + w["Previous_Scores"] * X["Previous_Scores"]
            + w["Tutoring_Sessions"] * X["Tutoring_Sessions"]
            + w["Physical_Activity"] * X["Physical_Activity"]
        )
        return vals.to_numpy(dtype=float)


# ---- Fixtures ----
@pytest.fixture
def attendance_data():
    # 12 total sessions, 9 present -> 75% attendance
    return {"present": 9, "total_sessions": 12}

@pytest.fixture
def assessment_rows():
    # 6 rows spanning subjects and assessment types
    return [
        {
            "assessment_title": "Quiz 1", "title": "Quiz 1",
            "alpha_identifier": "ALG-1", "session_date": datetime(2025, 1, 10),
            "pre": True, "mid": False, "post": False,
            "subject": "Algebra", "score": 80, "max_score": 100,
            "sports_hours": 1, "tutor_sessions": 2, "study_hours": 3
        },
        {
            "assessment_title": "Quiz 2", "title": "Quiz 2",
            "alpha_identifier": "ALG-1", "session_date": datetime(2025, 1, 20),
            "pre": False, "mid": True, "post": False,
            "subject": "Algebra", "score": 90, "max_score": 100,
            "sports_hours": 2, "tutor_sessions": 3, "study_hours": 4
        },
        {
            "assessment_title": "Unit Test", "title": "Unit Test",
            "alpha_identifier": "ALG-1", "session_date": datetime(2025, 2, 1),
            "pre": False, "mid": False, "post": True,
            "subject": "Algebra", "score": 70, "max_score": 100,
            "sports_hours": 0.5, "tutor_sessions": 2, "study_hours": 2
        },
        {
            "assessment_title": "Quiz A", "title": "Quiz A",
            "alpha_identifier": "GEO-2", "session_date": datetime(2025, 1, 12),
            "pre": True, "mid": False, "post": False,
            "subject": "Geometry", "score": 50, "max_score": 80,
            "sports_hours": 0, "tutor_sessions": 1, "study_hours": 1
        },
        {
            "assessment_title": "Quiz B", "title": "Quiz B",
            "alpha_identifier": "GEO-2", "session_date": datetime(2025, 1, 25),
            "pre": False, "mid": True, "post": False,
            "subject": "Geometry", "score": 60, "max_score": 80,
            "sports_hours": 1, "tutor_sessions": 2, "study_hours": 2
        },
        {
            "assessment_title": "Quiz C", "title": "Quiz C",
            "alpha_identifier": "GEO-2", "session_date": datetime(2025, 2, 5),
            "pre": False, "mid": False, "post": True,
            "subject": "Geometry", "score": 70, "max_score": 80,
            "sports_hours": 2, "tutor_sessions": 3, "study_hours": 3
        },
    ]


# ---- Basic structure/utility tests ----
def test_empty_checks():
    aa = AssessmentAnalysis([], {})
    assert aa.isDataEmpty() is True
    assert aa.isAttendanceDataEmpty() is True
    assert aa.dataSize() == 0
    assert aa.data_() == []


def test_datasets(assessment_rows, attendance_data):
    aa = AssessmentAnalysis(assessment_rows, attendance_data)
    assert aa.isDataEmpty() is False
    assert aa.dataSize() == 6

    vals = aa.get_dataset_()
    assert pytest.approx(vals[0], rel=1e-6) == 80.0
    assert pytest.approx(vals[3], rel=1e-6) == (50/80)*100.0
    assert len(vals) == 6

    labels = aa.get_dataset_labels_()
    assert labels[:3] == ["Quiz 1", "Quiz 2", "Unit Test"]
    assert len(labels) == 6


def test_get_dataset_assessment(assessment_rows, attendance_data):
    aa = AssessmentAnalysis(assessment_rows, attendance_data)
    out = aa.get_dataset_assessment_()
    # one entry per alpha_identifier with pre/mid/post keys
    # expect ALG-1 and GEO-2
    keys = {r["alpha_identifier"] for r in out}
    assert {"ALG-1", "GEO-2"} <= keys
    # ensure the dicts have pre/mid/post
    for r in out:
        assert set(["pre", "mid", "post"]).issubset(r.keys())


def test_get_dataset_subjects(assessment_rows, attendance_data):
    aa = AssessmentAnalysis(assessment_rows, attendance_data)
    subj = aa.get_dataset_subjects_()
    assert "Algebra" in subj and "Geometry" in subj
    assert all(isinstance(v, list) for v in subj.values())
    assert all(isinstance(x, float) for v in subj.values() for x in v)


def test_moving_averages(assessment_rows, attendance_data):
    aa = AssessmentAnalysis(assessment_rows, attendance_data)
    frame = aa.assessment_moving_average_()
    assert set(frame.keys()) == {"SMA", "EMA", "CMA"}
    assert len(frame["SMA"]) == 6 and len(frame["EMA"]) == 6 and len(frame["CMA"]) == 6
    # SMA starts with zeros due to window=5
    assert frame["SMA"][0] == 0


def test_subject_moving_average_bias(assessment_rows, attendance_data):
    aa = AssessmentAnalysis(assessment_rows, attendance_data)
    out = aa.subject_moving_average_bias_()
    # returns list of {"subject": ..., "percent_change": ..., "mean": ...}
    assert all({"subject", "percent_change", "mean"} <= set(d.keys()) for d in out)
    # at least two subjects summarized
    subjects = {d["subject"] for d in out}
    assert {"Algebra", "Geometry"} <= subjects


def test_assessment_moving_average_subject(assessment_rows, attendance_data):
    aa = AssessmentAnalysis(assessment_rows, attendance_data)
    out = aa.assessment_moving_average_subject_()
    # [{ 'SMA:Algebra': value }, { 'SMA:Geometry': value }]
    keys = [list(d.keys())[0] for d in out]
    assert any(k.startswith("SMA:") for k in keys)
    assert len(out) >= 2


# ---- LR-based methods: model present vs missing ----
def test_assessment_analysis_lr_returns_none_when_missing_model(assessment_rows, attendance_data, tmp_path, monkeypatch):
    # No ./Models/linear_model.pkl in tmp dir
    monkeypatch.chdir(tmp_path)
    aa = AssessmentAnalysis(assessment_rows, attendance_data)
    assert aa.assessment_analysis_lr_() is None
    assert aa.assessment_analysis_lr_subject_() is None


def test_assessment_analysis_lr_happy_path(assessment_rows, attendance_data, tmp_path, monkeypatch):
    # Write dummy model into ./Models/linear_model.pkl
    models_dir = tmp_path / "Models"
    models_dir.mkdir()
    with open(models_dir / "linear_model.pkl", "wb") as f:
        pickle.dump(DummyLinearModel(), f)

    monkeypatch.chdir(tmp_path)

    aa = AssessmentAnalysis(assessment_rows, attendance_data)

    preds = aa.assessment_analysis_lr_()
    assert isinstance(preds, list)
    assert len(preds) == len(assessment_rows)
    for row in preds:
        assert {"prediction", "actual", "title"} <= set(row.keys())
        assert isinstance(row["prediction"], float)

    subj_preds = aa.assessment_analysis_lr_subject_()
    # [{'LR_<subject>': [values...]}, ...]
    assert isinstance(subj_preds, list) and len(subj_preds) >= 2
    # check that keys start with LR_ and values are lists of floats
    k0, v0 = list(subj_preds[0].items())[0]
    assert k0.startswith("LR_")
    assert all(isinstance(x, float) for x in v0)


# ---- Guard-rail tests for empty inputs ----
def test_methods_return_none_when_empty_data(attendance_data):
    aa = AssessmentAnalysis([], attendance_data)
    assert aa.get_dataset_() is None
    assert aa.get_dataset_labels_() is None
    assert aa.get_dataset_assessment_() is None
    assert aa.get_dataset_subjects_() is None
    assert aa.assessment_moving_average_() is None
    assert aa.subject_moving_average_bias_() is None
    assert aa.assessment_moving_average_subject_() is None
    assert aa.assessment_analysis_lr_() is None
    assert aa.assessment_analysis_lr_subject_() is None


def test_methods_return_none_when_empty_attendance(assessment_rows):
    aa = AssessmentAnalysis(assessment_rows, {})
    assert aa.assessment_analysis_lr_() is None
    assert aa.assessment_analysis_lr_subject_() is None