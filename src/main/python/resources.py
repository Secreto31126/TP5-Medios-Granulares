import os.path as pth

_base_path = None

def set_base_path(base: str):
    """
    Sets a custom base path for resource files.

    :param base: The custom base directory path.
    """
    global _base_path
    _base_path = base

def path(*name: str):
    """
    Returns the absolute path to the Java resource file.

    Assumes the resources are located in the 'resources' directory,
    one level up the cwd, unless a custom base path is set via set_base_path().

    :param name: The name of the resource file.
    :return: The absolute path to the resource file.
    """
    if _base_path is not None:
        return pth.abspath(pth.join(_base_path, *name))
    return pth.abspath(pth.join(pth.dirname(__file__), '..', '..', '..', 'sim', *name))
