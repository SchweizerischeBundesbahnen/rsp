def catch_zero_division_error_as_minus_one(_lambda):
    try:
        return _lambda()
    except ZeroDivisionError:
        return -1
