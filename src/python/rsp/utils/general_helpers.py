def catch_zero_division_error_as_minus_one(_lambda, ret=-1):
    try:
        return _lambda()
    except ZeroDivisionError:
        return ret
