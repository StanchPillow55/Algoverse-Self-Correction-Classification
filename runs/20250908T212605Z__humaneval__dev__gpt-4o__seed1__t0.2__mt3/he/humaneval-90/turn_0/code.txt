def next_smallest(lst):
    if len(lst) < 2:
        return None
    unique_sorted = sorted(set(lst))
    if len(unique_sorted) < 2:
        return None
    return unique_sorted[1]