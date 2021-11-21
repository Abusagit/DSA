import numpy as np
from itertools import combinations
from functools import reduce
import sys


class Node:
    def __init__(self, letter, distance=None, parent=None, left=None, right=None,
                 cumulative_distance=0):
        self.letter = letter
        self.distance = distance
        self.parent = parent
        self.left = left
        self.right = right
        self.cumulative_distance = cumulative_distance

    def is_leaf(self):
        return all((not self.right, not self.left))

    # def __repr__(self):
    #     return f"Left: {self.left.letter}, Right: {self.right.letter}, Parent: {self.parent.letter}, DIST: {self.distance}, CUMDIST: {self.cumulative_distance}"


class PhylTree:
    LETTERS = "ABCDEFGHIJKLMN"

    def __init__(self, score=None, leaves_amount=None):
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
            new_distance = matrix[i, j] / 2
            letter_i = num2letter[i]
            letter_j = num2letter[j]
            new_node_name = "{}{}".format(letter_i, letter_j)

            self.nodes[letter_i].distance = new_distance - self.nodes[letter_i].cumulative_distance
            self.nodes[letter_j].distance = new_distance - self.nodes[letter_j].cumulative_distance

            cumulative_distance = new_distance
            self.nodes[new_node_name] = Node(letter=new_node_name, left=self.nodes[letter_i],
                                             right=self.nodes[letter_j],
                                             cumulative_distance=cumulative_distance)

            self.nodes[letter_i].parent = self.nodes[letter_j].parent = self.nodes[new_node_name]

            temp_dist_dict = {}
            for node_name_1, node_name_2 in combinations(set(letter2num) - {letter_i, letter_j}, 2):
                temp_dist_dict[(node_name_1, node_name_2)] = temp_dist_dict[(node_name_2, node_name_1)] = matrix[
                    letter2num[node_name_1],
                    letter2num[node_name_2]
                ]

            cluster_i = 1 if not clusters else len(letter_i)
            cluster_j = 1 if not clusters else len(letter_j)
            cluster_size = cluster_i + cluster_j

            for letter in set(letter2num) - {letter_i, letter_j}:
                temp_dist_dict[(letter, new_node_name)] = temp_dist_dict[(new_node_name, letter)] = (
                        (cluster_i * matrix[letter2num[letter], letter2num[letter_i]] + cluster_j * matrix[
                            letter2num[letter], letter2num[letter_j]]) / cluster_size)

            new_letters = set(letter2num) - {letter_i, letter_j} | {new_node_name}
            letter2num = dict(zip(new_letters, range(len(new_letters))))
            num2letter = dict(zip(range(len(new_letters)), new_letters))

            score_array = [temp_dist_dict[(num2letter[i], num2letter[j])] for i in range(n - 1)
                           for j in range(i + 1, n - 1)]
            matrix = self._build_and_fill_weight_matrix(n - 1, score_array)

            n -= 1
        else:
            self.root = self.nodes[new_node_name]

    def upgma(self, score):
        self.wpgma(score=score, clusters=True)

    def neighbor_joining(self, sequences):
        seq2num = {self.LETTERS[i]: i for i in range(self.leaves_amount)}
        num2seq = {i: self.LETTERS[i] for i in range(self.leaves_amount)}
        n = self.leaves_amount

        D = self._build_initial_nj_matrix(n, sequences, num2seq)
        Q = self._buid_score_nj_matrix(n, D)

        while n > 2:
            i, j = np.unravel_index(np.argmin(matrix, axis=None), matrix.shape)

        else:
            # TODO make root from 2 left nodes

    @staticmethod
    def _build_and_fill_weight_matrix(n, score):
        upper_triang_indices = np.triu_indices(n, 1)
        M = np.eye(n,n) * 999999
        M[upper_triang_indices] = score
        M += M.T
        return M

    @staticmethod
    def _build_initial_nj_matrix(n, sequences, num2seq):
        M = np.zeros((n,n))
        for row in range(n):
            for col in range(n+1, n):
                M[row, col] = M[col, row] = PhylTree._hamming_distance(num2seq[row], num2seq[col])

        return M

    @staticmethod
    def _hamming_distance(seq_1, seq_2):
        return reduce(lambda x, y: x + (seq_1[y] != seq_2[y]), range(len(seq_1)), 0)

    @staticmethod
    def _buid_score_nj_matrix(n, D):
        Q = np.zeros((n,n))
        for row in range(n):
            for col in range(n):
                Q[row, col] = Q[col, row] = (n - 2) * D[row, col] - D[[row, col], :] @ np.ones(n) @ np.ones(2)

        return Q

    def _print_newick_notation(self, node: Node) -> str:
        if node.is_leaf():
            return f"{node.letter}:{node.distance:.2f}"
        string_left = self._print_newick_notation(node.left)
        string_right = self._print_newick_notation(node.right)
        return f"({string_left}, {string_right}){f':{node.distance:.2f}' if node.distance is not None else ''}"

    def __str__(self):
        string = self._print_newick_notation(self.root)
        return string


def test():
    # A = PhylTree(leaves_amount=5, score=[17, 21, 31, 23, 30, 34, 21, 28, 39, 43])
    # A = PhylTree(leaves_amount=4, score=[16, 16, 10, 8, 8, 4])
    A = PhylTree(leaves_amount=5, score=[5, 9, 9, 8, 10, 10, 9, 8, 7, 3])
    A.wpgma(A.score)
    print(A.nodes)
    print(A)


def homework_finite_machine_wpgma(input_string):
    leaves_num, algorithm, *score = input_string.strip().split(' ')
    leaves_num = int(leaves_num)
    score = list(map(float, score))
    tree = PhylTree(score=score, leaves_amount=leaves_num)
    algorithms = {"WPGMA": tree.wpgma, "UPGMA": tree.upgma}

    algorithms[algorithm](score=score)
    print(tree)

def finite_machine_nj(input_string):
    seq_num, *sequences = input_string.strip().split(' ')

    tree = PhylTree(leaves_amount=int(seq_num))
    print(tree.neighbor_joining(sequences=sequences))



if __name__ == '__main__':
    # homework_finite_machine_wpgma("5 WPGMA 20 2 2 4 13 10 16 2 2 21")
