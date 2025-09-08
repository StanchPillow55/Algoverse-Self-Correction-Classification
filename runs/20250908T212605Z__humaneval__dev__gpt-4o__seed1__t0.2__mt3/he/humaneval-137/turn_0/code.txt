def compare_one(a, b):
    def convert_to_float(value):
        if isinstance(value, str):
            value = value.replace(',', '.')
        return float(value)
    
    a_float = convert_to_float(a)
    b_float = convert_to_float(b)
    
    if a_float > b_float:
        return a
    elif b_float > a_float:
        return b
    else:
        return None