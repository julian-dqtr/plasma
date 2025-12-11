import os
import zipfile
import glob


def prepare_raw_data(
    zip_file_path: str, target_dir_path: str
) -> list[tuple[tuple[int, int], str]]:
    """extract iamges from a zipfile.

    Args:
      zip_file_path: path to the zip file containing all the plasma images.
      target_dir_path: target path to the directory (can be created if it odes not already exists).
    """
    os.makedirs(target_dir_path, exist_ok=True)
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(target_dir_path)
    image_paths = glob.glob(
        os.path.join(target_dir_path, "**", "*.tif"), recursive=True
    )

    def get_physics_from_path(path: str) -> tuple[int, int]:
        """files are stored in directories names X_Y_Z where Y and Z are physics measurements

        Args:
          path: file_path of the form a/directory/file.tif

        Returns:
          tuple of the two physics measures (tension and frequency)
        """
        dir_name = os.path.dirname(path)
        splits = dir_name.split("_")
        return int(splits[-2]), int(splits[-1])

    return [(get_physics_from_path(path), path) for path in image_paths]
