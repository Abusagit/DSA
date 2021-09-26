import numpy as np


class Alignment:
    def __init__(self, sequence_1, sequence_2, alphabet="AGCT"):
        self.sequence_1 = sequence_1
        self.sequence_2 = sequence_2
        self.alphabet = alphabet

    @staticmethod
    def _make_alignment_matrix(m, n, mismatch):
        alignment_matrix = np.zeros((m, n), dtype=int)
        alignment_matrix[0, 1:] = np.array([mismatch * i for i in range(1, n)])
        alignment_matrix[1:, 0] = np.array([mismatch * j for j in range(1, m)])
        return alignment_matrix

    def align_needleman_wunsch(self, match=1, mismatch=-1, gap=-1):
        n = len(self.sequence_1) + 1
        m = len(self.sequence_2) + 1
        upper_alignment = []
        lower_alignment = []

        decisions = {
            0: lambda col, row_: (self.sequence_1[col - 1], self.sequence_2[row_ - 1], col - 1, row_ - 1),  # MATCH/MISMATCH
            1: lambda col, row_: (self.sequence_1[col - 1], "_", col - 1, row_),  # gap in lower sequence
            2: lambda col, row_: ("_", self.sequence_2[row_ - 1], col, row_ - 1),  # gap in upper sequence
        }

        backward_pass = {(0, 0): 0}  # (decision

        for i in range(n - 1, 0, -1):
            backward_pass[(0, i)] = 2

        for j in range(m - 1, 0, -1):
            backward_pass[(j, 0)] = 1

        alignment_matrix = self._make_alignment_matrix(m, n, mismatch)

        for row in range(1, m):
            for column in range(1, n):
                if_MATCH = self.sequence_1[column - 1] == self.sequence_2[row - 1]
                prev_steps = (alignment_matrix[row - 1, column - 1] + match * if_MATCH + mismatch * (not if_MATCH),
                              alignment_matrix[row - 1, column] + gap,
                              alignment_matrix[row, column - 1] + gap,
                              )

                alignment_matrix[row, column] = np.max(prev_steps).astype(int)
                decision = np.argmax(prev_steps)

                backward_pass[row, column] = decision

        upper_symbol, lower_symbol, i, j = decisions[decision](column, row)
        while (i, j) in backward_pass:
            upper_alignment.append(upper_symbol)
            lower_alignment.append(lower_symbol)
            upper_symbol, lower_symbol, i, j = decisions[backward_pass[(i, j)]](i, j)

        return f"{''.join(reversed(upper_alignment))} {''.join(reversed(lower_alignment))}"

    def get_score_of_alignment(self):
        pass

    def alignment_by_matrix(self, matrix: np.array):
        pass


if __name__ == '__main__':
    alignment_ = Alignment("TACGTTCAAG", "TCAAG")
    print(alignment_.align_needleman_wunsch())
