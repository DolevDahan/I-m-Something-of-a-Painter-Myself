import os
import json

def load_mifid_from_json(folder):
    """
    Load MiFID scores from all JSON files in the given folder.

    Args:
        folder (str): Path to the folder containing JSON files.

    Returns:
        list: List of MiFID scores (floats).
    """
    mifid_scores = []

    for filename in os.listdir(folder):
        if filename.endswith('.json'):
            path = os.path.join(folder, filename)
            with open(path, 'r') as f:
                data = json.load(f)
                if 'MiFID' in data:
                    mifid_scores.append(data['MiFID'])

    return mifid_scores
