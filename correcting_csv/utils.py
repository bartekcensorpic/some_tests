from typing import (Callable, Iterable, TypeVar, Any, List)
from pathlib import Path
from tqdm import tqdm

R = TypeVar('R')

def process_files_generator(function_to_use: Callable[[str], R], folder_path:str, pattern: Iterable, tqdm_description:str = None)-> R:
    """
    Recursively walks through FOLDER_PATH, collects all files that match extension and perform FUNCTION_TO_USE on each file. Yields an object of whatever FUNCTION_TO_USE returns

    :param tqdm_description: Description to be displayed witch each iteration
    :param function_to_use: Function to be performed on each file
    :param folder_path: Folder where files will be searched
    :param pattern: List of strings with pattern to match against files in folder
    :return: yields whatever FUNCTION_TO_USE returns
    """
    files = []
    for ex in pattern:
        files.extend(Path(folder_path).rglob(ex))

    pbar = tqdm(files)

    for file_path in pbar:
        pbar.set_description(tqdm_description)

        f_result = function_to_use(str(file_path))
        yield f_result


