def validate_key_is_nested_one_layer(key: str) -> bool:
    """
    Check if the given key is nested only one layer deep.

    A key is considered nested one layer deep if it contains at most one dot ('.') character.

    Parameters:
    ----------
        key (str): The key to validate.

    Returns:
        bool: True if the key is nested only one layer deep, False otherwise.
    """
    return key.count(".") <= 1
