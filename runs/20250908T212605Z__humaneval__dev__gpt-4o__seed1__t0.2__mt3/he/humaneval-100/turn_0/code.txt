def make_a_pile(n):
    pile = []
    current_stones = n
    for _ in range(n):
        pile.append(current_stones)
        if current_stones % 2 == 0:
            current_stones += 2
        else:
            current_stones += 2
    return pile