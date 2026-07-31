from searchkernel.compression import truncate_delta


def test_truncate_delta_no_truncation():
    """Test truncate_delta with input under max_lines."""
    short_diff = "\n".join([f"line {i}" for i in range(50)])
    result = truncate_delta(short_diff, max_lines=200)
    assert result == short_diff
    assert "omitted" not in result


def test_truncate_delta_with_truncation():
    """Test truncate_delta with input exceeding max_lines."""
    long_diff = "\n".join([f"line {i}" for i in range(300)])
    result = truncate_delta(long_diff, max_lines=200)
    assert "lines omitted" in result
    lines = result.splitlines()
    # Should have 200 lines + empty line + omission message
    assert len([line for line in lines if line and "omitted" not in line]) == 200
