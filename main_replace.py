from pathlib import Path
from joblib import Parallel, delayed
import multiprocessing
from pathlib import Path
import functools


def inplace_change(filename, old_string, new_string):
    with open(filename) as f:
        newText = f.read().replace(old_string, new_string)

    with open(filename, "w") as f:
        f.write(newText)


def replace_in_file(file_path, mappings):
    file_path = str(file_path)

    for old_string, new_string in mappings.items():
        inplace_change(file_path, old_string, new_string)

    print(f"done file: {file_path}")


def replace_names(folder_name):

    num_cores = multiprocessing.cpu_count()
    files = list(Path(folder_name).rglob(r"*.xml"))

    mappings = {
        "nipples": "IT WAS NIIPLES",
        "Naked torso of women": "nakedWoman",
        "Naked torso of men": "nakedMan",
    }

    print(f"doing folder {folder}")
    results = Parallel(n_jobs=num_cores)(
        delayed(replace_in_file)(file, mappings) for file in files
    )

    print(f"done {folder_name}")


if __name__ == "__main__":
    FOLDERS = [
        r"C:\Users\barte\Documents\Projects\some_tests\files",
    ]

    for folder in FOLDERS:
        replace_names(folder)

    print("script done")
