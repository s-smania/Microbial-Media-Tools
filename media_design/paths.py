from pathlib import Path

# Data files shipped with the package live in media_design/data/
DATA_DIR = Path(__file__).resolve().parent / "data"


def data_path(name) -> Path:
    """
    Return the path to a data file bundled with this package.

    Parameters
    ----------
    name : str
        File name of a data file shipped in the package 'data' directory,
        e.g. 'master-reagents.csv'.

    Returns
    -------
    pathlib.Path
        Absolute path to the bundled data file. The file is not required to
        exist; no validation is performed here.
    """
    return DATA_DIR / name


def resolve_data_path(path=None, default=None) -> Path:
    """
    Resolve a user-supplied file path, falling back to the bundled data directory.

    Parameters
    ----------
    path : str or path-like or None, optional
        Path supplied by the user. If it points to an existing file it is used
        as-is. If it does not exist but its file name matches a file shipped in
        the package 'data' directory, the bundled file is used instead. This
        allows a bare name such as 'master-reagents.csv' to work regardless of
        the current working directory.
    default : str or None, optional
        File name of the bundled data file to use when `path` is None or empty.

    Returns
    -------
    pathlib.Path
        Path to use for reading. If neither the user path nor a bundled file
        exists, the user path is returned unchanged so that the resulting
        error message refers to what the user asked for.

    Raises
    ------
    ValueError
        If `path` is None or empty and no `default` was given.

    Notes
    -----
    - An existing user path always wins; the bundled copy is only a fallback.
    - Only the file name of `path` is used when looking in the data directory,
      so 'data/master-reagents.csv' and 'master-reagents.csv' both resolve.
    """
    if path is None or (isinstance(path, str) and not path.strip()):
        if default is None:
            raise ValueError("No path given and no default data file specified.")
        return data_path(default)

    path = Path(path)

    if path.exists():
        return path

    bundled = data_path(path.name)
    if bundled.exists():
        return bundled

    return path
