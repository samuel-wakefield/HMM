import os
import glob
from typing import List, Dict


def load_dice_data(path: str) -> List[Dict[str, List[str]]]:
    """
    Loads the dice dataset from the path

    @param path: path to the dice data folder
    @return: A list of dictionaries with a field ('observed') for the observed sequence and one for the hidden sequence ('hidden'), each encoded as a list of strings.
    Observations are encoded as a string: '1', '2', ..., '6'
    Hidden states are encoded as a string: 'F': FAIR, 'W': WEIGHTED;
    """
    dice_data = []
    dice_files = glob.glob(os.path.join(path, "*"))
    dice_files.sort()
    for file in dice_files:
        with open(file, encoding="utf-8") as f:
            content = f.readlines()
            observed = content[0].strip()
            hidden = content[1].strip()
            dice_data.append({"observed": list(observed), "hidden": list(hidden)})
    return dice_data
