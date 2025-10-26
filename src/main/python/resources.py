import os.path as pth

def path(*name: str):
    """
    Returns the absolute path to the Java resource file.

    Assumes the resources are located in the 'resources' directory,
    one level up the cwd.

    :param name: The name of the resource file.
    :return: The absolute path to the resource file.
    """
    return pth.abspath(pth.join(pth.dirname(__file__), '..', '..', '..', 'sim', *name))
