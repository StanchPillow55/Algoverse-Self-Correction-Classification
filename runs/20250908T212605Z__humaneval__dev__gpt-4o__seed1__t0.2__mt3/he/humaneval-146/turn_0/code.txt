def specialFilter(nums):
    def is_odd_digit(digit):
        return digit in {'1', '3', '5', '7', '9'}
    
    count = 0
    for num in nums:
        if num > 10:
            str_num = str(num)
            if is_odd_digit(str_num[0]) and is_odd_digit(str_num[-1]):
                count += 1
    return count