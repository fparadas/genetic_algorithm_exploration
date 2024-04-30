import pytest
from ga.types.result import Result


def test_result_ok():
    """Test the Result.Ok creates a result with a value and no error."""
    value = 123
    result = Result.Ok(value)
    assert result.is_ok(), "Result should be okay"
    assert result.value == value, "Result value should match the input"
    assert result.error is None, "Result should not have an error"


def test_result_error():
    """Test the Result.Error creates a result with an error and no value."""
    error_message = "An error occurred"
    result = Result.Error(error_message)
    assert result.is_error(), "Result should be an error"
    assert result.error == error_message, "Result error should match the input"
    assert result.value is None, "Result should not have a value"


def test_result_not_both():
    """Test that a Result cannot be both Ok and Error."""
    with pytest.raises(ValueError):
        Result(value=123, error="An error occurred")


def test_result_neither():
    """Test that a Result must be either Ok or Error, not neither."""
    with pytest.raises(ValueError):
        Result()


def test_match():
    """Test the __match_args__ attribute of the Result class."""
    res = Result.Ok(123)
    match res:
        case Result(value) if value is not None:
            assert value == 123
        case Result(error) if error is not None:
            raise AssertionError("Should not match an error")
    err = Result.Error("Failure occurred")
    match err:
        case Result(value) if value is not None:
            raise AssertionError("Should not match a value")
        case Result(error) if error is not None:
            assert error == "Failure occurred"
