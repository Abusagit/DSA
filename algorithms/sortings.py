import random
import heapq


def check_sorted(array, ascending=True):
    """
    Tests whether array is sorted or not
    O(N)

    int(True) = 1 --> +1
    int(False) = 0 --> -1
    x*2-1 <--- flag for ascending or descending check
    """
    N = len(array)
    for i in range(N - 1):
        if array[i] > array[i + 1] if ascending else array[i] < array[i + 1]:
            return False
    return True


def selectionSort(array):
    def find_smallest(array):
        smallest = array[0]
        smallest_index = 0
        for i in range(1, len(array)):
            if array[i] < smallest:
                smallest = array[i]
                smallest_index = i
        return smallest_index

    new_array = []
    for i in range(len(array)):
        smallest = find_smallest(array)
        new_array.append(array.pop(smallest))
    return new_array


def merge_arrays(left, right):
    """
    Performs merging of 2 sorted arrays
    """
    M = [None] * (len(left) + len(right))
    # print(C)
    i = k = n = 0
    while i < len(left) and k < len(right):  # Пока не вышли за границы массива
        if left[i] <= right[k]:
            M[n] = left[i]
            i += 1
        else:
            M[n] = right[k]
            k += 1
        n += 1  # left or right is out of range here
    # while i < len(left):
    #     M[n] = left[i]
    #     i += 1
    #     n += 1
    M[n:] = left[i:]
    n += len(left) - i
    # while k < len(right):
    #     M[n] = right[k]
    #     k += 1
    #     n += 1
    M[n:] = right[k:]
    # print(C)
    return M


def quickSort(a):
    """

    :param a:
    :return:
    """
    if len(a) < 2:
        return a
    else:
        # print('pivot', pivot)
        pivot = random.randint(0, len(a) - 1)
        middle = [i for i in a if i == a[pivot]]
        less = [i for i in a if i < a[pivot]]
        # print('less', less)
        greater = [i for i in a if i > a[pivot]]
        # print('greater', greater)
        return quickSort(less) + middle + quickSort(greater)


# def partition_stable(a, l, r):
#     x, j, t = a[l], l, r
#     i = j
#     while i <= t:
#         if a[i] < x:
#             a[j], a[i] = a[i], a[j]
#             j += 1
#
#         elif a[i] > x:
#             a[t], a[i] = a[i], a[t]
#             t -= 1
#             i -= 1  # remain in the same i in this case
#         i += 1
#     return j, t
#
#
# def choarSort(a, l, r):
#     if l >= r:
#         return
#     k = random.randint(l, r)
#     a[l], a[k] = a[k], a[l]
#     m1, m2 = partition_stable(a, l, r)
#     choarSort(a, l, m1 - 1)
#     choarSort(a, m2 + 1, r)


def merge_sort(a):
    """
    param a: array

    Performs sorting by  merging
    """
    if len(a) <= 1:
        return a
    middle = len(a) // 2
    L = merge_sort(a[:middle])
    R = merge_sort(a[middle:])
    # print(L, R)
    return merge_arrays(L, R)


def heapsort(array):
    """
    :param array:
    :return:
    """
    heapq.heapify(array)
    return [heapq.heappop(array) for i in range(len(array))]


def insert_sort(array):
    """Sort by inserts"""
    for top in range(1, len(array)):
        k = top
        while k > 0 and array[k - 1] > array[k]:
            array[k], array[k - 1] = array[k - 1], array[k]
            k -= 1


def choise_sort(array):
    """sort by choises"""
    for pos in range(len(array)):
        for k in range(pos + 1, len(array)):
            if array[k] < array[pos]:
                array[pos], array[k] = array[k], array[pos]


def bubble_sort(array):
    """sort by Bubble-sorting method"""
    for bypass in range(1, len(array)):
        for k in range(len(array) - bypass):
            if array[k] > array[k + 1]:
                array[k], array[k + 1] = array[k + 1], array[k]


def countingSort(array, place=1, in_place=False):
    output = [0 for _ in range(len(array))]
    _b = [0 for _ in range(max(array) + 1)]
    for i in array:
        _b[i] += 1
    for i in range(1, len(_b)):
        _b[i] += _b[i - 1]
    print(_b)
    i = len(array) - 1
    while i >= 0:
        output[_b[array[i]] - 1] = array[i]
        _b[array[i]] -= 1
        i -= 1
    if in_place:
        for i in range(0, len(array)):
            array[i] = output[i]
    return output


def _count_sort(arr, place):
    size = len(arr)
    output = [0 for _ in range(size)]
    count = [0 for _ in range(10)]  # Decimal system

    for i in range(size):
        index = arr[i] // place
        count[index % 10] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    i = size - 1
    while i >= 0:
        index = arr[i] // place
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1

    for i in range(size):
        arr[i] = output[i]


def radixSort(array):
    max_element = max(array)
    place = 1
    while max_element // place > 0:
        _count_sort(array, place)
        place *= 10
