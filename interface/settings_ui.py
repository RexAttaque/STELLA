"""Fields of the algorithm settings (config/settings.py): on the pages (main settings) and in the Settings window
(Settings > Algorithm settings... in the top menu bar). Every field is bound to the store: editing it sets the value,
and a value set elsewhere (results file loaded, reset) shows in the field."""
import dearpygui.dearpygui as dpg

from config.settings import BY_KEY, PAGES, SETTINGS, STORE, user_settings_path

WINDOW_TAG = "settings_window"
_listening = False


def _tag(key: str) -> str:
    return f"setting_{key}"


def _on_store_change(key, value) -> None:
    if dpg.does_item_exist(_tag(key)):
        dpg.set_value(_tag(key), value)
    if dpg.does_item_exist(WINDOW_TAG):
        _refresh_changed_marks()


def _ensure_listener() -> None:
    global _listening
    if not _listening:
        STORE.add_listener(_on_store_change)
        _listening = True


def _on_field(sender, app_data, user_data) -> None:
    v = STORE.set(user_data, app_data)
    if dpg.get_value(sender) != v:  # clamped
        dpg.set_value(sender, v)


def add_setting_field(key: str, width: int = 150, info_theme=None) -> None:
    """Input field (with a '?' help popup) of setting `key`, in the current container."""
    _ensure_listener()
    s = BY_KEY[key]
    with dpg.group(horizontal=True):
        common = dict(label=s.label, width=width, tag=_tag(key), user_data=key, callback=_on_field, on_enter=True)
        if s.is_int:
            dpg.add_input_int(default_value=int(STORE.get(key)), step=1, **common)
        else:
            # (double: the value typed reaches the store exactly, a float field would round it to float32)
            dpg.add_input_double(default_value=float(STORE.get(key)), format=s.fmt, step=0, **common)
        btn = dpg.add_button(label="?", width=22, height=22)
        if info_theme is not None:
            dpg.bind_item_theme(btn, info_theme)
        with dpg.popup(btn, mousebutton=dpg.mvMouseButton_Left):
            dpg.add_text(f"{s.help}\n\nDefault: {s.default:{'d' if s.is_int else 'g'}}", wrap=380)


def _refresh_changed_marks() -> None:
    changed = STORE.changed()
    for s in SETTINGS:
        mark = f"setting_mark_{s.key}"
        if dpg.does_item_exist(mark):
            dpg.set_value(mark, "*" if s.key in changed else " ")
    if dpg.does_item_exist("settings_changed_text"):
        n = len(changed)
        dpg.set_value("settings_changed_text",
                      f"{n} setting(s) differ from the defaults (marked *)." if n else "All settings at their defaults.")


def _reset(page=None, window_only=True) -> None:
    for s in SETTINGS:
        if (page is None or s.page == page) and (not window_only or not s.main):
            STORE.set(s.key, s.default)


def build_settings_window(info_theme=None) -> None:
    """The Settings window (hidden until opened from the menu)."""
    _ensure_listener()
    if dpg.does_item_exist(WINDOW_TAG):
        return
    with dpg.window(tag=WINDOW_TAG, label="Algorithm settings", show=False, width=680, height=760, pos=(560, 60)):
        dpg.add_text("Settings that change the results but are rarely tuned. The main ones are on their pages "
                     "(listed under each section). Every value is kept for the next session; detection results "
                     "files also store the detection values.", wrap=640, color=(180, 180, 180))
        dpg.add_text("", tag="settings_changed_text", color=(255, 200, 80))
        with dpg.group(horizontal=True):
            dpg.add_button(label="Reset all to defaults (page fields too)", callback=lambda: _reset(None, False))
        for page in PAGES:
            in_window = [s for s in SETTINGS if s.page == page and not s.main]
            on_page = [s for s in SETTINGS if s.page == page and s.main]
            with dpg.collapsing_header(label=page, default_open=True):
                if on_page:
                    dpg.add_text(f"On the {page} page: " + ", ".join(s.label for s in on_page) + ".",
                                 wrap=640, color=(150, 150, 150))
                group = None
                for s in in_window:
                    if s.group != group:
                        group = s.group
                        dpg.add_spacer(height=4)
                        dpg.add_text(group, color=(200, 200, 100))
                    with dpg.group(horizontal=True):
                        dpg.add_text(" ", tag=f"setting_mark_{s.key}", color=(255, 200, 80))
                        add_setting_field(s.key, width=150, info_theme=info_theme)
                if in_window:
                    dpg.add_button(label=f"Reset the {page} settings above", user_data=page,
                                   callback=lambda sd, a, u: _reset(u, True))
                dpg.add_spacer(height=6)
        dpg.add_separator()
        dpg.add_text(f"Saved in {user_settings_path()}", wrap=640, color=(130, 130, 130))
    _refresh_changed_marks()


def show_settings_window() -> None:
    if dpg.does_item_exist(WINDOW_TAG):
        _refresh_changed_marks()
        dpg.configure_item(WINDOW_TAG, show=True)
        dpg.focus_item(WINDOW_TAG)
