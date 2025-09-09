def valid_date(date):
    if not date:
        return False
    
    if len(date) != 10 or date[2] != '-' or date[5] != '-':
        return False
    
    try:
        month, day, year = map(int, date.split('-'))
    except ValueError:
        return False
    
    if month < 1 or month > 12:
        return False
    
    if month in {1, 3, 5, 7, 8, 10, 12}:
        if day < 1 or day > 31:
            return False
    elif month in {4, 6, 9, 11}:
        if day < 1 or day > 30:
            return False
    elif month == 2:
        if day < 1 or day > 29:
            return False
    
    return True