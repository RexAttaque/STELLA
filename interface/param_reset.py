"""Reset of the parameters to their defaults: a circular-arrow button next to every parameter field, and
Reset > <step> in the menu bar for all the parameters of a step (its page fields and its Settings window fields).

install() runs once the pages are built: it records the value of every parameter field as its default (algorithm
settings: their default in config/settings.py, not a value loaded from the user's settings file) and adds the
buttons. A reset sets the value and calls the field's own callback, as typing it would (events filtered again,
method groups shown, settings store updated, ...). Defaults that depend on the data are computed when reset: the
number of time steps of the detection and the time-crop range.
"""
import inspect

import dearpygui.dearpygui as dpg
import numpy as np

from config.settings import BY_KEY

PAGE_LABELS = {"home": "Preprocessing", "detection": "Detection", "tracking": "Tracking", "validation": "Validation"}
SETTING_PAGE = {"Detection": "detection", "Tracking": "tracking", "Validation": "validation", "Loading": "home"}

_PARAM_TYPES = ("mvInputInt", "mvInputFloat", "mvInputDouble", "mvCombo", "mvCheckbox", "mvSliderInt",
                "mvSliderFloat", "mvDragInt", "mvDragFloat", "mvRadioButton")
# fields that are not parameters with a default: file formats of the save boxes, the sensor size (read from the file)
_SKIP = {"save_filtered_format", "det_save_results_format", "track_save_results_format", "vali_save_results_format",
         "bracket_preprocessing_width", "bracket_preprocessing_height"}
# names of the fields without a label
_NAMES = {"preprocessing_polarity": "polarity", "det_method_combo": "detection method",
          "track_method_combo": "tracking method"}
_TEXTURE = "param_reset_icon"
_THEME = "param_reset_theme"

_entries = []  # dicts: page, label, changed(), reset()
_tabs = {}


# ---------------------------------------------------------------------------------------------- icon
def _icon_rgba(size=16, ss=4):
    """Circular arrow (open circle with an arrowhead), anti-aliased by supersampling."""
    n = size * ss
    yy, xx = (np.mgrid[0:n, 0:n] + 0.5) / ss - size / 2
    r = np.hypot(xx, yy)
    ang = np.degrees(np.arctan2(-yy, xx)) % 360
    radius, half = size * 0.30, size * 0.085
    gap = 55  # degrees left open, from 0 (3 o'clock) counter-clockwise
    ring = (np.abs(r - radius) <= half) & ~((ang > 0) & (ang < gap))
    # clockwise arrow: the head sits at the `gap` end of the ring and points into the gap
    a = np.radians(gap)
    tip_c = np.array([np.cos(a), -np.sin(a)]) * radius
    tangent = np.array([np.sin(a), np.cos(a)])  # screen direction of decreasing angle (clockwise)
    normal = np.array([np.cos(a), -np.sin(a)])
    w = size * 0.24
    p0 = tip_c + tangent * w * 1.1
    p1 = tip_c - tangent * w * 0.3 + normal * w
    p2 = tip_c - tangent * w * 0.3 - normal * w

    def side(pa, pb):
        return (xx - pa[0]) * (pb[1] - pa[1]) - (yy - pa[1]) * (pb[0] - pa[0])

    s0, s1, s2 = side(p0, p1), side(p1, p2), side(p2, p0)
    head = ((s0 >= 0) & (s1 >= 0) & (s2 >= 0)) | ((s0 <= 0) & (s1 <= 0) & (s2 <= 0))
    cov = (ring | head).reshape(size, ss, size, ss).mean(axis=(1, 3))
    rgba = np.zeros((size, size, 4), np.float32)
    rgba[..., :3] = 0.85
    rgba[..., 3] = cov
    return rgba.ravel()


def _ensure_resources():
    if not dpg.does_item_exist(_TEXTURE):
        if not dpg.does_item_exist("texture_registry_main"):
            dpg.add_texture_registry(tag="texture_registry_main")
        dpg.add_static_texture(16, 16, _icon_rgba(), tag=_TEXTURE, parent="texture_registry_main")
    if not dpg.does_item_exist(_THEME):
        with dpg.theme(tag=_THEME):
            with dpg.theme_component(dpg.mvImageButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0, 0))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (90, 150, 200, 120))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (50, 110, 160, 160))
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)


# ---------------------------------------------------------------------------------------------- reset
def _fire(item, value):
    """Sets the field and calls its callback as Dear PyGui would (sender, app_data, user_data)."""
    dpg.set_value(item, value)
    cb = dpg.get_item_callback(item)
    if cb is None:
        return
    args = (item, value, dpg.get_item_user_data(item))
    try:
        params = inspect.signature(cb).parameters.values()
        if any(p.kind == p.VAR_POSITIONAL for p in params):
            n = 3
        else:
            n = sum(1 for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))
    except (TypeError, ValueError):
        n = 3
    cb(*args[:min(n, 3)])


def _same(a, b, single=False):
    """Equal values; single: compared as the float32 a float field stores."""
    if isinstance(a, float) or isinstance(b, float):
        try:
            return np.float32(a) == np.float32(b) if single else float(a) == float(b)
        except (TypeError, ValueError):
            return False
    return a == b


def _decimal(v):
    """The value typed in a float32 field (0.4 for 0.4000000059604645): its shortest float32 representation."""
    return float(str(np.float32(v)))


def _field_entry(page, item, default, post=None):
    """default: value, or callable giving it at reset time; post: called after the reset."""
    single = dpg.get_item_type(item).endswith("mvInputFloat") or dpg.get_item_type(item).endswith("mvSliderFloat")         or dpg.get_item_type(item).endswith("mvDragFloat")

    def value():
        return default() if callable(default) else default

    def reset():
        v = value()
        if not _same(dpg.get_value(item), v, single):
            _fire(item, v)
        if post is not None:
            post()

    alias = dpg.get_item_alias(item)
    return {"page": page, "label": dpg.get_item_label(item) or _NAMES.get(alias, alias) or str(item),
            "changed": lambda: not _same(dpg.get_value(item), value(), single), "reset": reset, "value": value}


def _add_button(entry, parent, before=0):
    btn = dpg.add_image_button(_TEXTURE, width=14, height=14, parent=parent, before=before,
                               callback=lambda: entry["reset"]())
    dpg.bind_item_theme(btn, _THEME)
    with dpg.tooltip(btn):
        dpg.add_text("", tag=f"{btn}_tip")

    def _tip():  # (the data-dependent defaults change)
        try:
            v = entry["value"]()
            dpg.set_value(f"{btn}_tip", "Reset to default" + (f": {v:g}" if isinstance(v, float) else f": {v}"))
        except Exception:
            dpg.set_value(f"{btn}_tip", "Reset to default")

    with dpg.item_handler_registry() as h:
        dpg.add_item_hover_handler(callback=lambda: _tip())
    dpg.bind_item_handler_registry(btn, h)
    return btn


def _place_after(entry, item):
    """The button right after the field: in its row when the field is in a horizontal group, otherwise the field is
    moved into a new row with the button."""
    parent = dpg.get_item_parent(item)
    if dpg.get_item_type(parent).endswith("mvGroup") and dpg.get_item_configuration(parent).get("horizontal"):
        sib = dpg.get_item_children(parent, 1)
        i = sib.index(item)
        return _add_button(entry, parent, before=sib[i + 1] if i + 1 < len(sib) else 0)
    row = dpg.add_group(horizontal=True, parent=parent, before=item)
    dpg.move_item(item, parent=row)
    return _add_button(entry, row)


# ---------------------------------------------------------------------------------------------- install
def _walk(item):
    for c in dpg.get_item_children(item, 1) or []:
        yield c
        yield from _walk(c)


def _special_defaults(page, alias):
    """(default, post) of the fields whose default depends on the data, else None."""
    pre, det = _tabs.get("home"), _tabs.get("detection")
    if alias == "det_input_matching_N" and det is not None:
        return det._compute_default_N, lambda: setattr(det, "_n_user_set", False)
    if alias in ("time_tmin", "time_tmax") and pre is not None:
        k = 0 if alias == "time_tmin" else -1
        build_value = _decimal(dpg.get_value(alias))
        return (lambda: float(pre.raw_events[2][k]) if getattr(pre, "raw_events", None) else build_value), None
    return None


def _install_q_matrix(tracking_tab):
    cells = [c for row in getattr(tracking_tab, "q_inputs", []) for c in row]
    if not cells or not dpg.does_item_exist("Q_info_btn"):
        return
    defaults = [dpg.get_value(c) for c in cells]

    def reset():
        for c, v in zip(cells, defaults):
            dpg.set_value(c, v)

    entry = {"page": "tracking", "label": "Q matrix", "reset": reset, "value": lambda: "diag(2e-6, 1, 4e5)",
             "changed": lambda: any(not _same(dpg.get_value(c), v) for c, v in zip(cells, defaults))}
    _entries.append(entry)
    _place_after(entry, dpg.get_alias_id("Q_info_btn"))


def install(tabs_by_key, settings_window=None):
    """Adds the reset buttons to the pages (tab_<key> items) and to the Settings window."""
    _ensure_resources()
    _tabs.update(tabs_by_key)
    roots = [(key, f"tab_{key}") for key in tabs_by_key]
    if settings_window and dpg.does_item_exist(settings_window):
        roots.append((None, settings_window))
    for page, root in roots:
        if not dpg.does_item_exist(root):
            continue
        masks_list = dpg.get_alias_id("masks_list") if dpg.does_item_exist("masks_list") else None
        fields = []
        for item in _walk(dpg.get_alias_id(root)):
            t = dpg.get_item_type(item).split("::")[-1]
            if t not in _PARAM_TYPES:
                continue
            alias = dpg.get_item_alias(item)
            label = dpg.get_item_label(item) or ""
            if alias in _SKIP or label.startswith("##Q_"):
                continue
            if masks_list is not None and _inside(item, masks_list):
                continue
            fields.append((item, alias))
        for item, alias in fields:  # (placing buttons changes the tree: walk first)
            p = page
            if alias.startswith("setting_"):
                s = BY_KEY[alias[len("setting_"):]]
                default, post = s.default, None
                p = SETTING_PAGE[s.page]
            else:
                special = _special_defaults(page, alias)
                if special:
                    default, post = special
                else:
                    default, post = dpg.get_value(item), None
                    if dpg.get_item_type(item).split("::")[-1] in ("mvInputFloat", "mvSliderFloat", "mvDragFloat"):
                        default = _decimal(default)  # (the field holds the typed default as float32)
            entry = _field_entry(p, item, default, post)
            _entries.append(entry)
            _place_after(entry, item)
    if "tracking" in tabs_by_key:
        _install_q_matrix(tabs_by_key["tracking"])


def _inside(item, ancestor):
    while item:
        if item == ancestor:
            return True
        item = dpg.get_item_parent(item)
    return False


def entries(page=None):
    return [e for e in _entries if page is None or e["page"] == page]


def reset_page(page) -> int:
    """Resets every parameter of a step that differs from its default; returns how many were reset."""
    changed = [e for e in entries(page) if e["changed"]()]
    for e in changed:
        e["reset"]()
    tab = _tabs.get(page)
    log = getattr(tab, "log_status", None) or getattr(tab, "_status_log_append", None)
    msg = (f"{PAGE_LABELS.get(page, page)} parameters reset to their defaults ({len(changed)} changed: "
           + ", ".join(e["label"] for e in changed) + ")" if changed
           else f"{PAGE_LABELS.get(page, page)} parameters already at their defaults")
    if log:
        log(msg)
    else:
        print(msg)
    return len(changed)


def build_menu():
    """Reset menu of the menu bar (call inside dpg.menu_bar)."""
    with dpg.menu(label="Reset"):
        for key, label in PAGE_LABELS.items():
            dpg.add_menu_item(label=f"Reset {label} parameters", user_data=key,
                              callback=lambda s, a, u: reset_page(u))
        dpg.add_separator()
        dpg.add_menu_item(label="Reset all steps", callback=lambda: [reset_page(k) for k in PAGE_LABELS])
