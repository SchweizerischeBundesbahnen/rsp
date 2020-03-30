def catch_zero_division_error_as_minus_one(l):
    try:
        return l()
    except ZeroDivisionError:
        return -1
