from pathlib import Path

def get_level(level_name) -> list:
    """
    Converts the contents of a file representing a grid to a nested list format.
    
    :param file_path: Path to the input file containing the grid.
    :return: A list of lists representing the grid.
    """
    file_path = Path(__file__).resolve().parent.parent / "Pacman_Level" / f"{level_name}.pml"
    level = []
    with open(file_path, 'r') as file:
        for line in file:
            # Strip the newline character and create a list of characters
            row = list(line.strip())
            # Skip empty lines
            if row:
                level.append(row)
    return level