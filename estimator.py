from utils import load_dice_data, print_matrices
import os
from model import estimate_hmm
import random
import math

from typing import List, Dict, Tuple


def viterbi(
    observed_sequence: List[str],
    transition_probs: Dict[Tuple[str, str], float],
    emission_probs: Dict[Tuple[str, str], float],
) -> List[str]:
    """
    Uses the Viterbi algorithm to calculate the most likely single sequence of hidden states given the observed sequence and a model. Use the same symbols for the start and end observations as in tick 7 ('B' for the start observation and 'Z' for the end observation).

    @param observed_sequence: A sequence of observed die rolls
    @param: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    @param: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    @return: The most likely single sequence of hidden states
    """

    observed_sequence = ["B"] + observed_sequence + ["Z"]

    start_state = "B"
    end_state = "Z"
    states = set([state_pair[0] for state_pair in transition_probs])

    sigma = {(state, 0): None for state in states}
    psi = {}

    sigma[(start_state, 0)] = 0

    # Calculates maximum value of sigma coming from each possible previous state
    def next_sigma(state, i, observation):
        max_sigma = -math.inf
        max_prev_state = ""
        for prev_state in states:
            potential_sigma = try_log(
                transition_probs[(prev_state, state)],
                emission_probs[(state, observation)],
                sigma[(prev_state, i - 1)],
            )
            if potential_sigma is not None and potential_sigma > max_sigma:
                max_sigma = potential_sigma
                max_prev_state = prev_state

        if max_sigma == -math.inf:
            return None, None
        else:
            return max_sigma, max_prev_state

    def try_log(a, b, c):
        return (math.log(a) + math.log(b) + c) if all([a, b, c is not None]) else None

    # Iterates through each observed state and constructs table of each possible path and its corresponding probability
    for i, observation in enumerate(observed_sequence):
        if i != 0:
            for state in states:
                sig, prev_state = next_sigma(state, i, observation)
                sigma[(state, i)] = sig
                psi[(state, i)] = prev_state

    # Starting at the most likely final state, backtracks to find most likely sequence of states
    seq = [end_state]
    current_state = end_state
    k = len(observed_sequence) - 1

    for i in range(k, 1, -1):
        current_state = psi[(current_state, i)]
        seq.append(current_state)

    seq = seq[::-1]

    return seq[:-1]


def precision_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the precision of the estimated sequence with respect to the positive class (weighted state), i.e. here the proportion of predicted weighted states that were actually weighted.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The precision of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    count = 0
    total = 0

    for pred_list, true_list in zip(pred, true):
        for pred_val, true_val in zip(pred_list, true_list):
            if pred_val == 1:
                total += 1
                if true_val == 1:
                    count += 1

    return count / total


def recall_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the recall of the estimated sequence with respect to the positive class (weighted state), i.e. here the proportion of actual weighted states that were predicted weighted.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The recall of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    count = 0
    total = 0

    for pred_list, true_list in zip(pred, true):
        for pred_val, true_val in zip(pred_list, true_list):
            if true_val == 1:
                total += 1
                if pred_val == 1:
                    count += 1

    return count / total


def f1_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the F1 measure of the estimated sequence with respect to the positive class (weighted state), i.e. the harmonic mean of precision and recall.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The F1 measure of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    p = precision_score(pred, true)
    r = recall_score(pred, true)

    return 0 if p * r == 0 else 2 * (p * r) / (p + r)


def main():
    dice_data = load_dice_data(os.path.join("data", "dice_dataset"))

    seed = 2
    print(
        f"Evaluating HMM on a single training and dev split using random seed {seed}."
    )
    random.seed(seed)
    dice_data_shuffled = random.sample(dice_data, len(dice_data))
    dev_size = int(len(dice_data) / 10)
    train = dice_data_shuffled[dev_size:]
    dev = dice_data_shuffled[:dev_size]
    dev_observed_sequences = [sequence["observed"] for sequence in dev]
    dev_hidden_sequences = [sequence["hidden"] for sequence in dev]
    predictions = []
    transition_probs, emission_probs = estimate_hmm(train)

    for sample in dev_observed_sequences:
        prediction = viterbi(sample, transition_probs, emission_probs)
        predictions.append(prediction)

    predictions_binarized = [
        [1 if pred_state == "W" else 0 for pred_state in pred] for pred in predictions
    ]
    dev_hidden_sequences_binarized = [
        [1 if state == "W" else 0 for state in sequence]
        for sequence in dev_hidden_sequences
    ]

    print("The transition probabilities of the HMM:")
    print_matrices(transition_probs)
    print("The emission probabilities of the HMM:")
    print_matrices(emission_probs)

    p = precision_score(predictions_binarized, dev_hidden_sequences_binarized)
    r = recall_score(predictions_binarized, dev_hidden_sequences_binarized)
    f1 = f1_score(predictions_binarized, dev_hidden_sequences_binarized)

    print(f"Precision for seed {seed} using the HMM: {p}")
    print(f"Recall for seed {seed} using the HMM: {r}")
    print(f"F1 for seed {seed} using the HMM: {f1}\n")


if __name__ == "__main__":
    main()
