"""Default locations of the file dialogs."""
import os
import sys
from pathlib import Path


def _windows_videos_dir():
    """The Videos known folder of the logged-in user (follows a relocation, e.g. to OneDrive), or None."""
    import ctypes
    from ctypes import wintypes
    import uuid

    folder_id = uuid.UUID("{18989B1D-99B5-455B-841C-AB7C74E4DDFC}")  # FOLDERID_Videos
    guid = (ctypes.c_byte * 16).from_buffer_copy(folder_id.bytes_le)
    path = ctypes.c_wchar_p()
    shell32 = ctypes.windll.shell32
    shell32.SHGetKnownFolderPath.argtypes = [ctypes.c_void_p, wintypes.DWORD, wintypes.HANDLE,
                                             ctypes.POINTER(ctypes.c_wchar_p)]
    if shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(path)) != 0:
        return None
    try:
        return Path(path.value)
    finally:
        ctypes.windll.ole32.CoTaskMemFree(path)


def default_files_dir() -> str:
    """Default folder of the file dialogs: the Videos folder of the logged-in user (home folder if it has none)."""
    videos = None
    if sys.platform == "win32":
        try:
            videos = _windows_videos_dir()
        except (OSError, AttributeError, ValueError):
            videos = None
    if videos is None:
        videos = Path(os.path.expanduser("~")) / "Videos"
    return str(videos if videos.is_dir() else Path(os.path.expanduser("~")))
