
def list_content_of_directory(directory: str) -> dict:
    """
    List the content of a directory

    :param directory: The directory to list
    :return: The content of the directory
    """
    import os

    entries = os.listdir(directory)
    files, dirs = [], []

    return {
        "files": [e for e in entries if os.path.isfile(os.path.join(directory, e))],
        "directories": [e for e in entries if os.path.isdir(os.path.join(directory, e))]
    }


def read_ascii_file(file: str) -> list:
    """
    Read an ASCII file

    :param file: The file to read
    :return: The content of the file
    """
    import numpy as np

    return np.loadtxt(file, skiprows=6).tolist()


def read_json(file: str) -> dict or list:
    """
    Read a JSON file

    :param file: The file to read
    :return: The content of the file
    """
    import json

    with open(file, 'r') as f:
        return json.load(f)