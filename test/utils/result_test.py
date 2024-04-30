import pytest
from utils.result import Result


def test_result_ok_unpacking():
    # Test unpacking of an Ok result
    val, err = Result.Ok(1)
    assert val == 1
    assert err is None


def test_result_error_unpacking():
    # Test unpacking of an Error result
    val, err = Result.Error("Error message")
    assert val is None
    assert err == "Error message"


def test_result_methods():
    # Test is_ok and is_error methods
    ok_result = Result.Ok(10)
    error_result = Result.Error("Failed")

    assert ok_result.is_ok()
    assert not ok_result.is_error()
    assert not error_result.is_ok()
    assert error_result.is_error()


# Running this test will validate that your Result class is functioning as intended
