def binary_search(lst, item):
    """improved search;
    dataset must be sorted!"""
    left = 0
    right = len(lst) - 1

    while left <= right:
        mid = left + (right - left) // 2
        if lst[mid] == item:
            return mid
        if lst[mid] > item:
            right = mid - 1
        else:
            left = mid + 1
    return -1, left


if __name__ == '__main__':
    my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(binary_search(my_list, 3))
    print(binary_search(my_list, 10))
    print(binary_search([1, 2], 1))
    print(binary_search([1, 2, 3, 5], 4))
    print(binary_search([1, 2, 3, 5], 6))
    a = [1, 2, 3, 5]
    a.insert(4, 6)
    print(a)
