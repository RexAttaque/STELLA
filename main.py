import sys

# Windows: hide the console window that opens with STELLA (double-click, shortcut, py launcher). It is only
# hidden when no shell shares it, so a terminal STELLA was started from stays visible. `--console` keeps it.
# The console still exists (prints and the joblib workers use it); it comes back if STELLA stops on an error.
_SHELLS = {"cmd.exe", "powershell.exe", "pwsh.exe", "bash.exe", "sh.exe", "wsl.exe", "code.exe"}


def _console_window():
    """Handle of this process's console window when Windows opened it for STELLA alone, else None."""
    if sys.platform != "win32" or "--console" in sys.argv[1:]:
        return None
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetConsoleWindow.restype = wintypes.HWND
        hwnd = k32.GetConsoleWindow()
        if not hwnd:
            return None
        pids = (wintypes.DWORD * 64)()
        n = k32.GetConsoleProcessList(pids, 64)
        if not 0 < n <= 64:
            return None
        k32.OpenProcess.restype = wintypes.HANDLE
        for pid in pids[:n]:
            h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return None  # unknown process on the console: leave it alone
            buf = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(1024)
            ok = k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size))
            k32.CloseHandle(h)
            if not ok or buf.value.rsplit("\\", 1)[-1].lower() in _SHELLS:
                return None
        return hwnd
    except Exception:
        return None


def _show(hwnd, visible: bool) -> None:
    import ctypes
    ctypes.windll.user32.ShowWindow(hwnd, 5 if visible else 0)  # SW_SHOW / SW_HIDE


_console = _console_window()
if _console:
    _show(_console, False)


def main() -> None:
    try:
        from core.app import run
        run()
    except Exception:
        if _console:  # show what went wrong instead of closing silently
            import traceback
            traceback.print_exc()
            _show(_console, True)
            input("STELLA stopped on an error. Press Enter to close.")
        raise


if __name__ == "__main__":
    main()
