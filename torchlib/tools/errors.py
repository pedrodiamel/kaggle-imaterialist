

class PYTVISIONError(Exception):
    """
    PYTVISION custom exception
    """
    pass


class DeleteError(PYTVISIONError):
    """
    Errors that occur when deleting a job
    """
    pass


class LoadImageError(PYTVISIONError):
    """
    Errors that occur while loading an image
    """
    pass


class UnsupportedPlatformError(PYTVISIONError):
    """
    Errors that occur while performing tasks in unsupported platforms
    """
    pass
