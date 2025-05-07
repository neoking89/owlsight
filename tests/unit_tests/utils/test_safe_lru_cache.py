import pytest
from owlsight.utils.helper_functions import safe_lru_cache

# Counter to track how many times a function is actually called
call_counter = 0

@safe_lru_cache(maxsize=128)
def function_with_list_param(items):
    """Test function that accepts a list parameter"""
    global call_counter
    call_counter += 1
    return len(items)

def test_safe_lru_cache_with_unhashable_types():
    """
    Test that safe_lru_cache properly handles unhashable types like lists
    without raising TypeError.
    """
    global call_counter
    call_counter = 0
    
    # First call with a list - should execute the function
    result1 = function_with_list_param(['a', 'b', 'c'])
    assert result1 == 3
    assert call_counter == 1
    
    # Second call with the same list - should successfully return from cache
    # even though lists are unhashable in standard lru_cache
    result2 = function_with_list_param(['a', 'b', 'c'])
    assert result2 == 3
    assert call_counter == 1, "Function should not be called again, result should be from cache"
    
    # Different list should execute the function again
    result3 = function_with_list_param(['a', 'b', 'c', 'd'])
    assert result3 == 4
    assert call_counter == 2


@safe_lru_cache(maxsize=128)
def function_that_raises_error(value):
    """Test function that raises an error for certain inputs"""
    global call_counter
    call_counter += 1
    if value < 0:
        raise ValueError("Value must be non-negative")
    return value * 2


def test_safe_lru_cache_error_handling():
    """
    Test that safe_lru_cache doesn't cache calls that raise exceptions,
    but still caches successful calls.
    """
    global call_counter
    call_counter = 0
    
    # Successful call - should be cached
    result1 = function_that_raises_error(5)
    assert result1 == 10
    assert call_counter == 1
    
    # Same input - should be retrieved from cache
    result2 = function_that_raises_error(5)
    assert result2 == 10
    assert call_counter == 1, "Function should not be called again for cached input"
    
    # Call that raises error
    with pytest.raises(ValueError):
        function_that_raises_error(-5)
    assert call_counter == 2, "Function should be called for input that raises error"
    
    # Retry same error-causing input - should still call function, not use cache
    with pytest.raises(ValueError):
        function_that_raises_error(-5)
    assert call_counter == 3, "Error-causing input should not be cached"
