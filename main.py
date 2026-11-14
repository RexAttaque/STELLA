import sys
import threading
import time

# Windows: hide the console window that opens with STELLA (double-click, shortcut, py launcher) and close it when
# STELLA ends. Only a console Windows opened for STELLA alone is touched: one shared with a shell (the terminal
# STELLA was started from) stays as it is. `--console` keeps it visible. The console itself stays attached while
# STELLA runs (prints and the joblib workers use it, no new windows); it comes back if STELLA stops on an error.
# When Windows Terminal hosts the console, GetConsoleWindow() is only a stand-in: the Terminal window owning it is
# the one hidden, and closed at the end (otherwise "closeOnExit": "never" would keep it open, hidden).
_SHELLS = {"cmd.exe", "powershell.exe", "pwsh.exe", "bash.exe", "sh.exe", "wsl.exe", "code.exe"}
_TERMINAL_CLASS = "CASCADIA_HOSTING_WINDOW_CLASS"


class _Console:
    window = None  # the window shown for the console (conhost window or Windows Terminal window)
    terminal = False  # hosted by Windows Terminal
    _hidden = False

    @classmethod
    def find(cls) -> None:
        if sys.platform != "win32" or "--console" in sys.argv[1:]:
            return
        try:
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            u32 = ctypes.WinDLL("user32", use_last_error=True)
            k32.GetConsoleWindow.restype = wintypes.HWND
            k32.OpenProcess.restype = wintypes.HANDLE
            u32.GetAncestor.restype = wintypes.HWND
            u32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
            u32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
            con = k32.GetConsoleWindow()
            if not con:
                return
            pids = (wintypes.DWORD * 64)()
            n = k32.GetConsoleProcessList(pids, 64)
            if not 0 < n <= 64:
                return
            for pid in pids[:n]:
                h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
                if not h:
                    return  # unknown process on the console: leave it alone
                buf = ctypes.create_unicode_buffer(1024)
                size = wintypes.DWORD(1024)
                ok = k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size))
                k32.CloseHandle(h)
                if not ok or buf.value.rsplit("\\", 1)[-1].lower() in _SHELLS:
                    return
            owner = u32.GetAncestor(con, 3)  # GA_ROOTOWNER
            name = ctypes.create_unicode_buffer(256)
            if owner and u32.GetClassNameW(owner, name, 256) and name.value == _TERMINAL_CLASS:
                cls.window, cls.terminal = owner, True
            else:
                cls.window = con
        except Exception:
            cls.window = None

    @classmethod
    def _show_window(cls, visible: bool) -> None:
        import ctypes
        from ctypes import wintypes
        u32 = ctypes.windll.user32
        u32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        u32.ShowWindow(cls.window, 5 if visible else 0)  # SW_SHOW / SW_HIDE

    @classmethod
    def hide(cls) -> None:
        """Hide now, and again for 3 s: Windows Terminal shows its window a moment after creating it."""
        if not cls.window:
            return
        cls._hidden = True
        cls._show_window(False)

        def keep_hidden():
            import ctypes
            from ctypes import wintypes
            u32 = ctypes.windll.user32
            u32.IsWindowVisible.argtypes = [wintypes.HWND]
            end = time.monotonic() + 3.0
            while cls._hidden and time.monotonic() < end:
                if u32.IsWindowVisible(cls.window):
                    cls._show_window(False)
                time.sleep(0.01)

        threading.Thread(target=keep_hidden, daemon=True, name="hide_console").start()

    @classmethod
    def show(cls) -> None:
        if cls.window:
            cls._hidden = False
            cls._show_window(True)

    @classmethod
    def close(cls) -> None:
        """Leave the console; close the Terminal window (it would stay with "closeOnExit": "never")."""
        if not cls.window:
            return
        cls._hidden = False
        import ctypes
        from ctypes import wintypes
        sys.stdout.flush()
        sys.stderr.flush()
        ctypes.windll.kernel32.FreeConsole()
        if cls.terminal:
            u32 = ctypes.windll.user32
            u32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            u32.PostMessageW(cls.window, 0x0010, 0, 0)  # WM_CLOSE


_Console.find()
_Console.hide()


def main() -> None:
    try:
        from core.app import run
        run()
    except Exception:
        if _Console.window:  # show what went wrong instead of closing silently
            import traceback
            traceback.print_exc()
            _Console.show()
            input("STELLA stopped on an error. Press Enter to close.")
            _Console.close()
        raise
    _Console.close()


if __name__ == "__main__":
    main()
