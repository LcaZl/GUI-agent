import pathlib
import argparse

def is_valid_file(path: str) -> str:
    """
    Identity function (i.e., the input is passed through and is not modified)
    that checks whether the given path is a valid file or not, raising an
    argparse.ArgumentTypeError if not valid.
    
    Parameters
    ----------
    path : str
        String representing a path to a file.
    
    Returns
    -------
    str
        The same string as in input
    
    Raises
    ------
    argparse.ArgumentTypeError
        An exception is raised if the given string does not represent a valid
        path to an existing file.
    """
    file = pathlib.Path(path)
    if not file.is_file():
        raise argparse.ArgumentTypeError(f"{path} does not exist")
    return path
