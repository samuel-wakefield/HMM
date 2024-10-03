from utils import load_dice_data, print_matrices
import os
from typing import List, Dict, Tuple


def get_transition_probs(
    hidden_sequences: List[List[str]],
) -> Dict[Tuple[str, str], float]:
    """
    Calculates the transition probabilities for the hidden dice types using maximum likelihood estimation. Counts the number of times each state sequence appears and divides it by the count of all transitions going from that state.

    @param hidden_sequences: A list of dice type sequences
    @return: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    """
    states = set([item for sublist in hidden_sequences for item in sublist])
    state_count = {state: 0 for state in states}
    transition_counts = {(s1, s2): 0 for s1 in states for s2 in states}

    for sequence in hidden_sequences:
        for i in range(len(sequence) - 1):
            current_state = sequence[i]
            next_state = sequence[i + 1]

            state_count[current_state] += 1
            transition_counts[(current_state, next_state)] += 1

    transition_probs = {}

    for (s1, s2), count in transition_counts.items():
        transition_probs[(s1, s2)] = 0 if count == 0 else count / state_count[s1]

    return transition_probs


def get_emission_probs(
    hidden_sequences: List[List[str]], observed_sequences: List[List[str]]
) -> Dict[Tuple[str, str], float]:
    """
    Calculates the emission probabilities from hidden dice states to observed dice rolls, using maximum likelihood estimation. Counts the number of times each dice roll appears for the given state (fair or loaded) and divides it by the count of that state.

    @param hidden_sequences: A list of dice type sequences
    @param observed_sequences: A list of dice roll sequences
    @return: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    """
    states = set([item for sublist in hidden_sequences for item in sublist])
    observations = set([item for sublist in observed_sequences for item in sublist])

    state_count = {state: 0 for state in states}

    observation_count = {
        (state, observation): 0 for state in states for observation in observations
    }

    for hidden_sequence, observed_sequence in zip(hidden_sequences, observed_sequences):
        for hidden_state, observed_state in zip(hidden_sequence, observed_sequence):
            state_count[hidden_state] += 1
            observation_count[(hidden_state, observed_state)] += 1

    emission_probs = {}

    for (state, observation), count in observation_count.items():
        emission_probs[(state, observation)] = (
            0 if count == 0 else count / state_count[state]
        )

    return emission_probs


def estimate_hmm(
    training_data: List[Dict[str, List[str]]],
) -> List[Dict[Tuple[str, str], float]]:
    """
    The parameter estimation (training) of the HMM. It consists of the calculation of transition and emission probabilities. Here 'B' is used for the start state and 'Z' for the end state, both for emissions and transitions.

    @param training_data: The dice roll sequence data (visible dice rolls and hidden dice types), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @return A list consisting of two dictionaries, the first for the transition probabilities, the second for the emission probabilities.
    """
    start_state = "B"
    end_state = "Z"
    observed_sequences = [
        [start_state] + sequence["observed"] + [end_state] for sequence in training_data
    ]
    hidden_sequences = [
        [start_state] + sequence["hidden"] + [end_state] for sequence in training_data
    ]
    transition_probs = get_transition_probs(hidden_sequences)
    emission_probs = get_emission_probs(hidden_sequences, observed_sequences)

    return [transition_probs, emission_probs]


def main():
    dice_data = load_dice_data(os.path.join("data", "dice_dataset"))
    transition_probs, emission_probs = estimate_hmm(dice_data)
    print("The transition probabilities of the HMM:")
    print_matrices(transition_probs)
    print("The emission probabilities of the HMM:")
    print_matrices(emission_probs)


if __name__ == "__main__":
    main()
