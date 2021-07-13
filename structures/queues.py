from collections import deque
import unittest


class Deque:
    def __init__(self):
        self.items = deque()

    def isEmpty(self):
        return self.size()

    def addFront(self, item):
        self.items.append(item)

    def addRear(self, item):
        self.items.appendleft(item)

    def removeFront(self):
        return self.items.pop()

    def removeRear(self):
        return self.items.popleft()

    def size(self):
        return len(self.items)


class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)


# this implementation of binary heap takes key value pairs,
# we will assume that the keys are all comparable
class PriorityQueue:
    def __init__(self):
        self.heapArray = [(0, 0)]
        self.currentSize = 0

    def __bool__(self):
        return len(self.heapArray) > 1

    def __repr__(self):
        return f'{self.heapArray[1:]}'

    def __len__(self):
        return len(self.heapArray) - 1  # subtract first supporting element

    def buildHeap(self, alist):
        self.currentSize = len(alist)
        self.heapArray = [(0, 0)]
        for i in alist:
            self.heapArray.append(i)
        i = len(alist) // 2
        while i > 0:
            self.percDown(i)
            i -= 1

    def percDown(self, i):
        while (i * 2) <= self.currentSize:
            mc = self.minChild(i)
            if self.heapArray[i][0] > self.heapArray[mc][0]:
                tmp = self.heapArray[i]
                self.heapArray[i] = self.heapArray[mc]
                self.heapArray[mc] = tmp
            i = mc

    def changePriority(self, index, new_prior):
        old_priority, self.heapArray[index][0] = self.heapArray[index][0], new_prior
        if old_priority > new_prior:
            self.percUp(index)
        else:
            self.percDown(index)

    def minChild(self, i):
        if i * 2 > self.currentSize:
            return -1
        else:
            if i * 2 + 1 > self.currentSize:
                return i * 2
            else:
                if self.heapArray[i * 2][0] < self.heapArray[i * 2 + 1][0]:
                    return i * 2
                else:
                    return i * 2 + 1

    def percUp(self, i):
        while i // 2 > 0:
            if self.heapArray[i][0] < self.heapArray[i // 2][0]:
                tmp = self.heapArray[i // 2]
                self.heapArray[i // 2] = self.heapArray[i]
                self.heapArray[i] = tmp
            i //= 2

    def add(self, k):
        self.heapArray.append(k)
        self.currentSize += 1
        self.percUp(self.currentSize)

    def delMin(self):
        retval = self.heapArray[1][1]
        self.heapArray[1] = self.heapArray[self.currentSize]
        self.currentSize -= 1
        self.heapArray.pop()
        self.percDown(1)
        return retval

    def isEmpty(self):
        return True if self.currentSize == 0 else False

    def decreaseKey(self, val, amt):
        # this is a little wierd, but we need to find the heap thing to decrease by
        # looking at its value
        done = False
        i = 1
        myKey = 0
        while not done and i <= self.currentSize:
            if self.heapArray[i][1] == val:
                done = True
                myKey = i
            else:
                i += 1
        if myKey > 0:
            self.heapArray[myKey] = (amt, self.heapArray[myKey][1])
            self.percUp(myKey)

    def __contains__(self, vtx):
        for pair in self.heapArray:
            if pair[1] == vtx:
                return True
        return False


class TestBinHeap(unittest.TestCase):
    def setUp(self):
        self.theHeap = PriorityQueue()
        self.theHeap.add((2, 'x'))
        self.theHeap.add((3, 'y'))
        self.theHeap.add((5, 'z'))
        self.theHeap.add((6, 'a'))
        self.theHeap.add((4, 'd'))

    def testInsert(self):
        assert self.theHeap.currentSize == 5

    def testDelmin(self):
        assert self.theHeap.delMin() == 'x'
        assert self.theHeap.delMin() == 'y'

    def testDecKey(self):
        self.theHeap.decreaseKey('d', 1)
        assert self.theHeap.delMin() == 'd'


if __name__ == '__main__':
    unittest.main()
