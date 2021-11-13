import numpy as np


class Alignment:
    def __init__(self, sequence_1, sequence_2, matrix_parameters, alphabet="ACGT"):
        self.sequence_1 = sequence_1
        self.sequence_2 = sequence_2
        self.alphabet = dict(zip(alphabet, range(len(alphabet))))
        self.matrix = build_weight_matrix(matrix_parameters, alphabet=alphabet)

    @staticmethod
    def _make_alignment_matrix(m, n, gap):
        alignment_matrix = np.zeros((m, n), dtype=int)
        alignment_matrix[0, 1:] = np.array([gap * i for i in range(1, n)])
        alignment_matrix[1:, 0] = np.array([gap * j for j in range(1, m)])
        return alignment_matrix

    @staticmethod
    def _make_backward_precursors(m, n):
        backward_pass = {(0, 0): 0}
        for i in range(n - 1, 0, -1):
            backward_pass[(0, i)] = 2

        for j in range(m - 1, 0, -1):
            backward_pass[(j, 0)] = 1

        return backward_pass

    def _decision(self, number, row, col):
        decisions = {
            0: lambda row, col: (self.sequence_1[col - 1], self.sequence_2[row - 1], row - 1, col - 1),
            # MATCH/MISMATCH
            1: lambda row, col: ("_", self.sequence_2[row - 1], row - 1, col),  # gap in upper sequence
            2: lambda row, col: (self.sequence_1[col - 1], "_", row, col - 1),  # gap in lower sequence
        }

        return decisions[number](row, col)

    def _compute_backward_pass(self, backward_dict, final_row, final_column, initial_decision):
        upper_sequence = []
        lower_sequence = []
        upper_symbol, lower_symbol, i, j = self._decision(initial_decision, final_row, final_column)
        while all((i >= 0, j >= 0)):
            upper_sequence.append(upper_symbol)
            lower_sequence.append(lower_symbol)
            upper_symbol, lower_symbol, i, j = self._decision(backward_dict[(i, j)], i, j)

        return upper_sequence, lower_sequence

    def align_needleman_wunsch(self, match=1, mismatch=-1, gap=-1):
        n = len(self.sequence_1) + 1
        m = len(self.sequence_2) + 1

        backward_pass = self._make_backward_precursors(m, n)  # decision

        alignment_matrix = self._make_alignment_matrix(m, n, gap)

        for row in range(1, m):
            for column in range(1, n):
                if_match = self.sequence_1[column - 1] == self.sequence_2[row - 1]
                prev_steps = (alignment_matrix[row - 1, column - 1] + match * if_match + mismatch * (not if_match),
                              alignment_matrix[row - 1, column] + gap,
                              alignment_matrix[row, column - 1] + gap,
                              )

                alignment_matrix[row, column] = np.max(prev_steps).astype(int)
                decision = np.argmax(prev_steps).astype(int)

                backward_pass[row, column] = decision

        upper_alignment, lower_alignment = self._compute_backward_pass(backward_pass, row, column, decision)
        return f"{''.join(reversed(upper_alignment))} {''.join(reversed(lower_alignment))}"

    def alignment_by_matrix(self, weight_matrix: np.ndarray, gap):
        n = len(self.sequence_1) + 1
        m = len(self.sequence_2) + 1

        backward_pass = self._make_backward_precursors(m, n)
        alignment_matrix = self._make_alignment_matrix(m, n, gap)

        for row in range(1, m):
            for column in range(1, n):
                code_for_seq_1_symbol = self.alphabet[self.sequence_1[column - 1]]
                code_for_seq_2_symbol = self.alphabet[self.sequence_2[row - 1]]

                prev_steps = (
                    alignment_matrix[row - 1, column - 1] + weight_matrix[code_for_seq_1_symbol, code_for_seq_2_symbol],
                    alignment_matrix[row - 1, column] + gap,
                    alignment_matrix[row, column - 1] + gap,
                )

                alignment_matrix[row, column] = np.max(prev_steps).astype(int)
                decision = np.argmax(prev_steps).astype(int)

                backward_pass[row, column] = decision

        upper_alignment, lower_alignment = self._compute_backward_pass(backward_pass, row, column, decision)
        return f"{''.join(reversed(upper_alignment))} {''.join(reversed(lower_alignment))}"

    @staticmethod
    def _make_alignment_tensor(m, n, gap_start_penalty, gap_prolong_penalty):
        alignment_tensor = np.ndarray(shape=(3, m, n), buffer=np.zeros((3, m, n)), dtype=int)

        alignment_tensor[0, 1, 0] = gap_start_penalty + gap_prolong_penalty  # TODO is this right?
        alignment_tensor[1, 1, 0] = alignment_tensor[2, 1, 0] = -9999

        for row in range(2, m):
            alignment_tensor[0, row, 0] = alignment_tensor[0, row - 1, 0] + gap_prolong_penalty
            alignment_tensor[1, row, 0] = -9999
            alignment_tensor[2, row, 0] = -9999

        alignment_tensor[2, 0, 1] = gap_start_penalty + gap_prolong_penalty  # TODO is this right?
        alignment_tensor[0, 0, 1] = alignment_tensor[1, 0, 1] = -9999
        for column in range(2, n):
            alignment_tensor[2, 0, column] = alignment_tensor[2, 0, column - 1] + gap_prolong_penalty
            alignment_tensor[1, 0, column] = -9999
            alignment_tensor[0, 0, column] = -9999

        return alignment_tensor

    @staticmethod
    def _make_backward_precursors_manhattan(m, n) -> np.ndarray:
        backward_decisions = np.ndarray(shape=(3, m, n), buffer=np.zeros((3, m, n)), dtype=int)

        # for i in range(3):
        #     backward_decisions[i, 0, 0] = 0
        #
        # for i in range(1, m):
        #     backward_decisions[0, i, 0] = 0

        for j in range(1, n):
            backward_decisions[2, 0, j] = 8

        return backward_decisions

    def _make_alignment_sequences(self, backward_decisions, initial_matrix_code, m, n):
        upper_alignment = []
        lower_alignment = []
        decision_code = backward_decisions[initial_matrix_code, -1, -1]
        upper_symbol, lower_symbol, matrix_code, row_idx, column_idx = self._decisions_manhattan(decision_code, m, n)

        while all((row_idx >= 0, column_idx >= 0)):
            upper_alignment.append(upper_symbol)
            lower_alignment.append(lower_symbol)
            decision_code = backward_decisions[matrix_code, row_idx, column_idx]
            upper_symbol, lower_symbol, matrix_code, row_idx, column_idx = self._decisions_manhattan(decision_code,
                                                                                                     row_idx,
                                                                                                     column_idx)

        return upper_alignment, lower_alignment

    def _decisions_manhattan(self, decision_code, row_index, column_index) -> tuple:
        decisions = {
            0: lambda row_idx, column_idx: ("_", self.sequence_2[row_idx - 1], 0, row_idx - 1, column_idx),
            1: lambda row_idx, column_idx: ("_", self.sequence_2[row_idx - 1], 1, row_idx - 1, column_idx),
            2: lambda row_idx, column_idx: ("_", self.sequence_2[row_idx - 1], 2, row_idx - 1, column_idx),
            3: lambda row_idx, column_idx: (
                self.sequence_1[column_idx - 1], self.sequence_2[row_idx - 1], 0, row_idx - 1, column_idx - 1),
            4: lambda row_idx, column_idx: (
                self.sequence_1[column_idx - 1], self.sequence_2[row_idx - 1], 1, row_idx - 1, column_idx - 1),
            5: lambda row_idx, column_idx: (
                self.sequence_1[column_idx - 1], self.sequence_2[row_idx - 1], 2, row_idx - 1, column_idx - 1),
            6: lambda row_idx, column_idx: (self.sequence_1[column_idx - 1], "_", 0, row_idx, column_idx - 1),
            7: lambda row_idx, column_idx: (self.sequence_1[column_idx - 1], "_", 1, row_idx, column_idx - 1),
            8: lambda row_idx, column_idx: (self.sequence_1[column_idx - 1], "_", 2, row_idx, column_idx - 1),
        }

        return decisions[decision_code](row_index, column_index)

    def affine_gaps_manhattan(self, gap_start_penalty, gap_prolong_penalty):
        m = len(self.sequence_2) + 1
        n = len(self.sequence_1) + 1
        alignment_tensor = self._make_alignment_tensor(m, n, gap_start_penalty, gap_prolong_penalty)

        backward_decisions = self._make_backward_precursors_manhattan(m, n)

        # for matrix_index in range(alignment_tensor.shape[0]):  # 3
        # 0 -> T, 1 -> S, 2 -> U
        for row_index in range(1, m):
            seq_2_symbol_code = self.alphabet[self.sequence_2[row_index - 1]]
            for column_index in range(1, n):
                seq_1_symbol_code = self.alphabet[self.sequence_1[column_index - 1]]
                steps_for_T = (
                    alignment_tensor[0, row_index - 1, column_index] + gap_prolong_penalty,
                    alignment_tensor[1, row_index - 1, column_index] + gap_start_penalty + gap_prolong_penalty,
                    alignment_tensor[2, row_index - 1, column_index] + gap_start_penalty + gap_prolong_penalty)
                weight_matrix_score = self.matrix[seq_1_symbol_code, seq_2_symbol_code]
                steps_for_S = (
                    alignment_tensor[0, row_index - 1, column_index - 1] + weight_matrix_score,
                    alignment_tensor[1, row_index - 1, column_index - 1] + weight_matrix_score,
                    alignment_tensor[2, row_index - 1, column_index - 1] + weight_matrix_score,)
                steps_for_U = (
                    alignment_tensor[0, row_index, column_index - 1] + gap_start_penalty + gap_prolong_penalty,
                    alignment_tensor[1, row_index, column_index - 1] + gap_start_penalty + gap_prolong_penalty,
                    alignment_tensor[2, row_index, column_index - 1] + gap_prolong_penalty,)

                alignment_tensor[0, row_index, column_index] = max(steps_for_T)
                backward_decisions[0, row_index, column_index] = np.argmax(steps_for_T)

                alignment_tensor[1, row_index, column_index] = max(steps_for_S)
                backward_decisions[1, row_index, column_index] = np.argmax(steps_for_S) + 3  # to create general 7 decisions

                alignment_tensor[2, row_index, column_index] = max(steps_for_U)
                backward_decisions[2, row_index, column_index] = np.argmax(steps_for_U) + 6

        max_score_matrix = np.argmax((alignment_tensor[0, -1, -1], alignment_tensor[1, -1, -1],
                                      alignment_tensor[2, -1, -1]))

        sequence_1_aligned, sequence_2_aligned = self._make_alignment_sequences(backward_decisions, max_score_matrix,
                                                                                m - 1, n - 1)

        alignment_score = alignment_tensor[max_score_matrix, -1, -1]
        return alignment_score, f"{''.join(reversed(sequence_1_aligned))} {''.join(reversed(sequence_2_aligned))}"

    def _decision_local(self, row_index, column_index, decision):
        decisions = {
            0: lambda row, col: (self.sequence_1[:col].lower(), self.sequence_2[:row].lower()),  # ??????
            1: lambda row, col: (self.sequence_1[col - 1], self.sequence_2[row - 1], row - 1, col - 1),
            # MATCH/MISMATCH
            2: lambda row, col: ("_", self.sequence_2[row - 1], row - 1, col),  # gap in upper sequence
            3: lambda row, col: (self.sequence_1[col - 1], "_", row, col - 1),  # gap in lower sequence
        }

        return decisions[decision](row_index, column_index)

    @staticmethod
    def _make_local_alignment_matrix_backward_decisions(m, n, gap):
        matrix = np.zeros((m, n), dtype=int)
        backward_decisions = np.zeros((m, n), dtype=int)

        backward_decisions[0, 0] = 0
        for i in range(1, m):
            possible_decisions = (0, float("-inf"), gap * i)  # 0 or 2 decision
            matrix[i, 0] = np.max(possible_decisions)
            backward_decisions[i, 0] = np.argmax(possible_decisions)

        for j in range(1, n):
            possible_decisions = (0, float("-inf"), float("-inf"), gap * j)  # 0 or 3 decision
            matrix[0, j] = np.max(possible_decisions)
            backward_decisions[0, j] = np.argmax(possible_decisions)

        return matrix, backward_decisions

    def _compute_backward_path_local(self, backward_decisions, max_row_index, max_column_index, initial_decision):
        upper_sub_sequence = []
        lower_sub_sequence = []

        decision = initial_decision
        row_index = max_row_index
        column_index = max_column_index
        while decision != 0:
            upper_symbol, lower_symbol, row_index, column_index = self._decision_local(row_index, column_index,
                                                                                       decision=decision)
            upper_sub_sequence.append(upper_symbol)
            lower_sub_sequence.append(lower_symbol)

            decision = backward_decisions[row_index, column_index]
        else:
            upper_residual, lower_residual = self._decision_local(row_index, column_index, decision)

        return (''.join(reversed(upper_sub_sequence)), ''.join(reversed(lower_sub_sequence)),
                upper_residual, lower_residual)

    def local_smith_waterman(self, gap_penalty):
        m = len(self.sequence_2) + 1
        n = len(self.sequence_1) + 1

        alignment_matrix, backward_decisions = self._make_local_alignment_matrix_backward_decisions(m, n, gap)

        for row_index in range(1, m):
            seq_2_symbol_code = self.alphabet[self.sequence_2[row_index - 1]]
            for column_index in range(1, n):
                seq_1_symbol_code = self.alphabet[self.sequence_1[column_index - 1]]
                weight_score = self.matrix[seq_1_symbol_code, seq_2_symbol_code]
                previous_decisions = (
                    0,
                    alignment_matrix[row_index-1, column_index-1] + weight_score,
                    alignment_matrix[row_index-1, column_index] + gap_penalty,  # gap in upper
                    alignment_matrix[row_index, column_index-1] + gap_penalty,  # gap in lower
                )

                alignment_matrix[row_index, column_index] = max(previous_decisions)

                current_decision = np.argmax(previous_decisions)
                backward_decisions[row_index, column_index] = current_decision

        max_row_idx, max_column_idx = np.unravel_index(np.argmax(alignment_matrix),
                                                       alignment_matrix.shape)  # row, column
        initial_decision = backward_decisions[max_row_idx, max_column_idx]

        alignment_score = alignment_matrix[max_row_idx, max_column_idx]

        upper_sub_sequence, lower_sub_sequence, upper_residual_left, lower_residual_left = self._compute_backward_path_local(
            backward_decisions,
            max_row_idx,
            max_column_idx,
            initial_decision)

        upper_residual_right = self.sequence_1[max_column_idx:].lower()
        lower_residual_right = self.sequence_2[max_row_idx:].lower()
        upper_alignment = ''.join((upper_residual_left, upper_sub_sequence, upper_residual_right))
        lower_alignment = ''.join((lower_residual_left, lower_sub_sequence, lower_residual_right))
        return alignment_score, f"{upper_alignment} {lower_alignment}"


def build_weight_matrix(array, alphabet="ACGT"):  # TODO for proteins
    size = {"ACGT": 4,
            "ACGU": 4,
            "amino": 20}
    if len(array) >= 4:
        return np.resize(array, new_shape=(size[alphabet], size[alphabet]))
    match, mismatch = array
    return match * np.eye(size[alphabet]) + mismatch * (np.ones(size[alphabet]) - np.eye(size[alphabet]))


if __name__ == '__main__':
    seq_1, seq_2, match_score, mismatch_score, open_gap_score, prolong_gap_score = "ATGC", "ATGC", -1, -1, 1, 0
    aligner = Alignment(seq_1, seq_2, (match_score, mismatch_score))
    print(*aligner.affine_gaps_manhattan(gap_start_penalty=open_gap_score, gap_prolong_penalty=prolong_gap_score))
    # seq_1, seq_2, match, mismatch, gap = "TGTTACGG", "GGTTGACTA", 1, -1, -1
    # aligner = Alignment(seq_1, seq_2, (match, mismatch))
    # print(*aligner.local_smith_waterman(gap))