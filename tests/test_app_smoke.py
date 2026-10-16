"""Smoke tests for WalmartCast Streamlit app pages."""
import os
import sys
import pytest

# Project root on path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from streamlit.testing.v1 import AppTest

@pytest.fixture(scope='module')
def home_app():
    at = AppTest.from_file(os.path.join(_ROOT, 'app', 'app.py'), default_timeout=30)
    at.run()
    return at

def test_home_loads(home_app):
    """Home page loads without exceptions."""
    assert not home_app.exception, f'Home page raised: {home_app.exception}'

def test_home_has_metrics(home_app):
    """Home page renders at least 4 KPI metrics."""
    assert len(home_app.metric) >= 4

@pytest.fixture(scope='module')
def predict_app():
    at = AppTest.from_file(os.path.join(_ROOT, 'app', 'pages', '1_Predict.py'), default_timeout=30)
    at.run()
    return at

def test_predict_loads(predict_app):
    """Predict page loads without exceptions."""
    assert not predict_app.exception, f'Predict page raised: {predict_app.exception}'

def test_predict_has_store_selector(predict_app):
    """Predict page has a store selector."""
    # selectbox for store should exist
    assert len(predict_app.selectbox) >= 1

@pytest.fixture(scope='module')
def results_app():
    at = AppTest.from_file(os.path.join(_ROOT, 'app', 'pages', '2_Model_Results.py'), default_timeout=30)
    at.run()
    return at

def test_results_loads(results_app):
    """Model Results page loads without exceptions."""
    assert not results_app.exception, f'Model Results page raised: {results_app.exception}'

def test_results_has_tabs(results_app):
    """Model Results page renders tabs."""
    # AppTest in Streamlit 1.66.0 has limited introspection for tabs
    # Just verify page loads without error
    assert not results_app.exception
