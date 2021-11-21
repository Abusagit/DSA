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
        letter2num = {self.LETTERS[i]: i for i in range(self.leaves_amount)}
        num2letter = {i: self.LETTERS[i] for i in range(self.leaves_amount)}
        n = self.leaves_amount

        D = self._build_initial_nj_matrix(n, sequences, num2letter)
        Q = self._buid_score_nj_matrix(n, D)

        try:
            while n > 2:
                i, j = np.unravel_index(np.argmin(Q), Q.shape)

                distance_i_new = D[i, j] / 2 + D[[i, j], :] @ np.ones(n) @ [1, -1] / (2 * n - 4)
                distance_j_new = D[i, j] - distance_i_new

                letter_i = num2letter[i]
                letter_j = num2letter[j]
                new_node_name = "{}{}".format(letter_i, letter_j)

                self.nodes[letter_i].distance = distance_i_new
                self.nodes[letter_j].distance = distance_j_new

                self.nodes[new_node_name] = Node(letter=new_node_name,
                                                 left=self.nodes[letter_i],
                                                 right=self.nodes[letter_j])

                self.nodes[letter_i].parent = self.nodes[letter_j].parent = self.nodes[new_node_name]

                temp_dist_dict = {}
                for node_name_1, node_name_2 in combinations(set(letter2num) - {letter_i, letter_j}, 2):
                    temp_dist_dict[(node_name_1, node_name_2)] = temp_dist_dict[(node_name_2, node_name_1)] = D[
                        letter2num[node_name_1],
                        letter2num[node_name_2]
                    ]

                for node_name in set(letter2num) - {letter_i, letter_j}:
                    temp_dist_dict[(node_name, new_node_name)] = temp_dist_dict[(new_node_name, node_name)] = (
                            1 / 2 * (D[i, letter2num[node_name]] + D[j, letter2num[node_name]] - D[i, j])
                    )

                new_letters = set(letter2num) - {letter_i, letter_j} | {new_node_name}
                letter2num = dict(zip(new_letters, range(len(new_letters))))
                num2letter = dict(zip(range(len(new_letters)), new_letters))
                score_array = [temp_dist_dict[(num2letter[i], num2letter[j])] for i in range(n - 1)
                               for j in range(i + 1, n - 1)]

                D = self._build_and_fill_weight_matrix_nj(n - 1, score_array)
                Q = self._buid_score_nj_matrix(n - 1, D)

                n -= 1

            else:
                letter_i = num2letter[0]
                letter_j = num2letter[1]

                self.nodes[letter_i].distance = self.nodes[letter_j].distance = temp_dist_dict[(letter_i, letter_j)] / 2

                self.root = Node(letter=f"{letter_i}{letter_j}",
                                 left=self.nodes[letter_i],
                                 right=self.nodes[letter_j])
                self.nodes[letter_i].parent = self.nodes[letter_j].parent = self.root

        except UnboundLocalError:
            if n == 2:
                distance = round(PhylTree._hamming_distance(*sequences) / 2, 2)
                self.root = Node(letter="AB",
                                 left=Node(
                                     letter="A",
                                     distance=distance
                                 ),
                                 right=Node(
                                     letter="B",
                                     distance=distance
                                 ))

                self.root.left.parent = self.root.right.parent = self.root

            if n == 1:
                self.root = Node(letter='A', distance=0)

    @staticmethod
    def _build_and_fill_weight_matrix_nj(n, score_array):
        D = np.zeros((n,n))
        u = np.triu_indices(n, 1)
        D[u] = score_array
        D += D.T
        return D

    @staticmethod
    def _build_and_fill_weight_matrix(n, score):
        upper_triang_indices = np.triu_indices(n, 1)
        M = np.eye(n, n) * 999999
        M[upper_triang_indices] = score
        M += M.T
        return M

    @staticmethod
    def _build_initial_nj_matrix(n, sequences, num2seq):  # TODO Redo
        M = np.zeros((n, n))
        for row in range(n):
            for col in range(row + 1, n):
                M[row, col] = M[col, row] = PhylTree._hamming_distance(sequences[row], sequences[col])

        return M

    @staticmethod
    def _hamming_distance(seq_1, seq_2):
        return sum(s_1 != s_2 for s_1, s_2 in zip(seq_1, seq_2))

    @staticmethod
    def _buid_score_nj_matrix(n, D):
        Q = np.eye(n) * 9999
        for row in range(n):
            for col in range(row + 1, n):
                Q[row, col] = Q[col, row] = (n - 2) * D[row, col] - D[[row, col], :] @ np.ones(n) @ np.ones(2)

        return Q

    def _print_newick_notation(self, node: Node) -> str:
        if node.is_leaf():
            return f"{node.letter}:{node.distance:.1f}"
        string_left = self._print_newick_notation(node.left)
        string_right = self._print_newick_notation(node.right)
        return f"({string_left}, {string_right}){f':{node.distance:.1f}' if node.distance is not None else ''}"

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
    tree.neighbor_joining(sequences=sequences)
    print(tree)


if __name__ == '__main__':
    # homework_finite_machine_wpgma("5 WPGMA 20 2 2 4 13 10 16 2 2 21")
    finite_machine_nj("2 TATAGATTT CATAGAGTG")
