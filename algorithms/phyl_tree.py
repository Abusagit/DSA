import numpy as np
from itertools import combinations
import sys


class Node:
    def __init__(self, letter, distance=None, parent=None, depth=-1, left=None, right=None, cumulative_distance=0,
                 clusters=False):
        self.letter = letter
        self.distance = distance
        self.parent = parent
        self.depth = depth
        self.left = left
        self.right = right
        self.cumulative_distance = cumulative_distance
        self.cluster_size = 1 if not clusters else len(self.letter)

    def is_leaf(self):
        return all((not self.right, not self.left))


class PhylTree:
    LETTERS = "ABCDEFGHIJKLMN"

    def __init__(self, score, leaves_amount):
        self.score = score
        self.root = None
        self.nodes = {leaf_letter: Node(letter=leaf_letter) for leaf_letter in self.LETTERS[:leaves_amount]}
        self.leaves_amount = leaves_amount

    def wpgma(self, score, clusters=False):
        letter2num = {self.LETTERS[i]: i for i in range(self.leaves_amount)}
        num2letter = {i: self.LETTERS[i] for i in range(self.leaves_amount)}
        n = self.leaves_amount
        matrix = self._build_and_fill_weight_matrix(n, score)

        while n >= 2:
            i, j = np.unravel_index(np.argmin(matrix, axis=None), matrix.shape)
            new_distance_i = new_distance_j = matrix[i, j] / 2
            letter_i = num2letter[i]
            letter_j = num2letter[j]
            new_node_name = "{}{}".format(letter_i, letter_j)

            self.nodes[letter_i].distance = new_distance_i - self.nodes[letter_i].cumulative_distance
            self.nodes[letter_j].distance = new_distance_j - self.nodes[letter_j].cumulative_distance

            self.nodes[new_node_name] = Node(letter=new_node_name, left=self.nodes[letter_i],
                                             right=self.nodes[letter_j],
                                             cumulative_distance=self.nodes[letter_j].cumulative_distance + new_distance_i)

            self.nodes[letter_i].parent = self.nodes[letter_j].parent = self.nodes[new_node_name]

            temp_dist_dict = {}
            for node_name_1, node_name_2 in combinations(set(letter2num) - {letter_i, letter_j}, 2):
                temp_dist_dict[(node_name_1, node_name_2)] = temp_dist_dict[(node_name_2, node_name_1)] = matrix[
                    letter2num[node_name_1],
                    letter2num[node_name_2]
                ]

            for letter in set(letter2num) - {letter_i, letter_j}:
                temp_dist_dict[(letter, new_node_name)] = temp_dist_dict[(new_node_name, letter)] = (
                    (matrix[letter2num[letter], letter2num[letter_i]] + matrix[letter2num[letter], letter2num[letter_j]])/2)

            new_letters = set(letter2num) - {letter_i, letter_j} | {new_node_name}
            letter2num = dict(zip(new_letters, range(len(new_letters))))
            num2letter = dict(zip(range(len(new_letters)), new_letters))

            score_array = [temp_dist_dict[(num2letter[i], num2letter[j])] for i in range(n - 1)
                           for j in range(i + 1, n - 1)]
            matrix = self._build_and_fill_weight_matrix(n - 1, score_array)

            n -= 1
        else:
            self.root = self.nodes[new_node_name]


    def neighbor_joining(self):
        pass

    @staticmethod
    def _build_and_fill_weight_matrix(n, score):
        upper_triang_indices = np.triu_indices(n, 1)
        lower_triang_indices = np.tril_indices(n, -1)
        M = np.zeros((n, n)) + 999999
        M[upper_triang_indices] = score
        M[lower_triang_indices] = score
        return M

    def _print_newick_notation(self, node: Node) -> str:
        if node.is_leaf():
            return f"{node.letter}:{node.distance}"
        string_left = self._print_newick_notation(node.left)
        string_right = self._print_newick_notation(node.right)
        return f"({string_left}, {string_right}){f':{node.distance:.2f}' if node.distance else ''}"

    def __str__(self):
        string = self._print_newick_notation(self.root)
        return string


def test():
    A = PhylTree(leaves_amount=5, score=[5, 9, 9, 8, 10, 10, 9, 8, 7, 3])
    A.wpgma(A.score)
    print(A)


def homework_finite_machine(input_string):
    leaves_num, algorithm, *score = input_string.strip().split(' ')
    leaves_num = int(leaves_num)
    score = list(map(float, score))
    tree = PhylTree(score=score, leaves_amount=leaves_num)
    algorithms = {"WPGMA": tree.wpgma, "UPGMA": tree.upgma}

    return algorithms[algorithm](score=score)


if __name__ == '__main__':
    test()

