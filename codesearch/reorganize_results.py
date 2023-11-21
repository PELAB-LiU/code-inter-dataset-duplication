import argparse
import glob
import os
import re
from pathlib import Path


def main(folder):
    folder = Path(folder)
    files = [f for f in os.listdir(folder) if os.path.isfile(folder / f)]
    files = [f for f in files if ".bin" in f or ".log" in f or ".pkl" in f]
    # get the model name
    file_names = [re.split("python", f)[0][:-1] for f in files]
    unique_file_names = list(set(file_names))

    for name in unique_file_names:
        (folder / name).mkdir(parents=True, exist_ok=True)

        # move files containing the name to the folder
        for file_name in files:
            if name == file_name.split("python")[0][:-1]:
                print(folder / name / file_name)
                os.rename(folder / file_name, folder / name / file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)

    args = parser.parse_args()

    main(args.folder)
