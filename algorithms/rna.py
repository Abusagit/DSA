import numpy as np
from collections import deque


class RNA:
    BASE_ENCODING = {
        "A": 1,
        "U": -1,
        "G": 2,
        "C": -2,
    }

    def __init__(self, string):
        self._string = string

    @staticmethod
    def _make_nussinov_matrix(size):
        return np.zeros((size, size))

    @classmethod
    def _is_complementary(cls, base_1, base_2):
        return not (cls.BASE_ENCODING[base_1] + cls.BASE_ENCODING[base_2])

    @staticmethod
    def _compute_backward_pass(backward_dict, initial_second_coordinate):
        stack = deque(((0, initial_second_coordinate),))
        estimated_connections = []

        while stack:
            i, j = stack.pop()
            backward_ancestors = backward_dict.get((i, j))
            if backward_ancestors:
                if len(backward_ancestors) == 1:
                    estimated_connections.append((i, j))

                stack.extend(backward_ancestors)

        return sorted(estimated_connections,key=lambda x: x[0])

    def estimate_secondary_sctructure(self, min_hairpin_size):
        n = len(self._string)
        nussinov_matrix = self._make_nussinov_matrix(n)

        backward_pass_array = {}
        for offset in range(1, n - min_hairpin_size):
            for j in range(min_hairpin_size + offset, n):
                i = j - min_hairpin_size - offset
                base_1 = self._string[j]
                base_2 = self._string[i]

                possible_combinations = tuple((
                    ((i, k), (k + 1, j)), nussinov_matrix[i, k] + nussinov_matrix[k + 1, j]) for k in range(i, j))

                max_from_previous_combinations = max(possible_combinations, key=lambda x: x[1])
                if self._is_complementary(base_1, base_2):
                    complementary_array_and_score = (((i + 1, j - 1),), nussinov_matrix[i + 1, j - 1] + 1)
                    max_score_transition = max((max_from_previous_combinations, complementary_array_and_score),
                                               key=lambda x: x[1])

                    backward_pass_array[(i, j)] = max_score_transition[0]
                    nussinov_matrix[i, j] = max_score_transition[1]
                # nussinov_matrix[i, j] = max(
                #     (nussinov_matrix[i + 1, j - 1] + 1) * self._is_complementary(base_1, base_2),
                #     max((nussinov_matrix[i, k] + nussinov_matrix[k + 1, j] for k in range(i, j)))
                # )
                else:
                    backward_pass_array[(i, j)] = max_from_previous_combinations[0]
                    nussinov_matrix[i, j] = max_from_previous_combinations[1]

        secondary_connections = self._compute_backward_pass(backward_pass_array, initial_second_coordinate=n - 1)

        return ' '.join(f"{i} {j}" for i, j in secondary_connections)


if __name__ == '__main__':
    a = RNA("AAGGGUUGGAAC")
    print(a.estimate_secondary_sctructure(min_hairpin_size=2))
