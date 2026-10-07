from natsort import natsorted


from pathlib import Path


def get_scene_names(folder_path):
    folder = Path(folder_path)

    # Find all .ply files in the folder
    ply_files = folder.glob("*.ply")

    # Extract filenames without extension
    names = [file.stem for file in ply_files]

    # Natural sort (e.g. file1, file2, ..., file10)
    sorted_names = natsorted(names)

    return sorted_names