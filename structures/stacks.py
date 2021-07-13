from collections import deque


class Stack:
    def __init__(self):
        self.items = deque()

    def isEmpty(self):
        return self.__len__()

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[-1]

    def __len__(self):
        return len(self.items)


class MaxStack(Stack):
    def __init__(self):
        super(MaxStack, self).__init__()
        self._max = float('-inf')

    def max(self):
        return self._max  # shows current maximum value in stack

    def push(self, item):
        if self.isEmpty():
            self._max = item
            super(MaxStack, self).push((item, float('-inf')))
        else:
            if item > self._max:
                super(MaxStack, self).push((item, self._max))
                self._max = item
            else:
                super(MaxStack, self).push((item, self._max))

    def pop(self):
        pop = super(MaxStack, self).pop()
        self._max = pop[1]
        return pop[0]
