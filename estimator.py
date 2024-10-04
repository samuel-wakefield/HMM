from utils import (
    load_dice_data,
    print_matrices,
    precision_score,
    recall_score,
    f1_score,
)
import os
from model import estimate_hmm
import random
import math

from typing import List, Dict, Tuple, Union


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


def cross_validation_sequence_labeling(
    data: List[Dict[str, List[str]]],
) -> Dict[str, float]:
    """
    Runs 10-fold cross-validation for evaluating the HMM's prediction with Viterbi decoding. Calculates precision, recall, and F1 for each fold and returns the average over the folds.

    @param data: the sequence data encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'
    @return: a dictionary with keys 'recall', 'precision', and 'f1' and its associated averaged score.
    """
    num_folds = 10
    folds = generate_random_cross_folds(data, num_folds)
    precisions = []
    recalls = []
    f1s = []

    # Iterate through each fold as the test fold
    for test_fold_index, test_fold in enumerate(folds):
        observed_sequences = [sequence["observed"] for sequence in test_fold]
        hidden_sequences = [sequence["hidden"] for sequence in test_fold]

        # Construct training data
        train = []
        for index, fold in enumerate(folds):
            if test_fold_index != index:
                train.extend(fold)

        predictions = []
        transition_probs, emission_probs = estimate_hmm(train)

        for sample in observed_sequences:
            prediction = viterbi(sample, transition_probs, emission_probs)
            predictions.append(prediction)

        predictions_binarized = [
            [1 if state == "W" else 0 for state in pred] for pred in predictions
        ]
        dev_hidden_sequences_binarized = [
            [1 if state == "W" else 0 for state in dev] for dev in hidden_sequences
        ]

        p = precision_score(predictions_binarized, dev_hidden_sequences_binarized)
        r = recall_score(predictions_binarized, dev_hidden_sequences_binarized)
        f1 = f1_score(predictions_binarized, dev_hidden_sequences_binarized)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f1 = sum(f1s) / len(f1s)

    return {"precision": avg_precision, "recall": avg_recall, "f1": avg_f1}


def generate_random_cross_folds(
    training_data: List[Dict[str, List[str]]], n: int
) -> List[List[Dict[str, List[str]]]]:
    """
    Split training data into n folds, random.

    @param training_data: list of training instances, where each instance is a list of dictionaries with each consisting of the fields 'observed', and 'hidden'
    @param n: the number of cross-folds
    @return: a list of n folds, where each fold is a list of training instances
    """
    shuffledIndexes = list(range(len(training_data)))
    random.shuffle(shuffledIndexes)

    folds = []
    fold_len = len(training_data) // n

    for fold_index in range(n):
        fold = []

        for fold_item in range(fold_index * fold_len, (fold_index + 1) * fold_len):
            fold.append(training_data[shuffledIndexes[fold_item]])

        folds.append(fold)

    return folds


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

    print("Evaluating HMM using cross-validation with 10 folds.\n")

    cv_scores = cross_validation_sequence_labeling(dice_data)

    print(f" Your cv average precision using the HMM: {cv_scores['precision']}")
    print(f" Your cv average recall using the HMM: {cv_scores['recall']}")
    print(f" Your cv average F1 using the HMM: {cv_scores['f1']}")


if __name__ == "__main__":
    main()
