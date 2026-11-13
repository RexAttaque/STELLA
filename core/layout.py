from typing import Sequence

import dearpygui.dearpygui as dpg
from config.constants import APP_TITLE, TAB_DEFINITIONS, THEME_COLORS, VIEWPORT_SIZE


class LayoutManager:
    def __init__(self, tabs: Sequence[object]) -> None:
        self.tabs = tabs
        self.main_window_tag = "main_window"
        self.detection_tab = next(t for t in self.tabs if t.__class__.__name__ == "DetectionTab")
        self._tabs_by_key = {d.key: t for d, t in zip(TAB_DEFINITIONS, self.tabs)}  # used by _on_tab_change

    def build(self) -> None:
        dpg.create_context()
        dpg.create_viewport(title=APP_TITLE, width=VIEWPORT_SIZE[0], height=VIEWPORT_SIZE[1])
        self._apply_theme()
        with dpg.window(label=APP_TITLE, tag=self.main_window_tag, width=VIEWPORT_SIZE[0], height=VIEWPORT_SIZE[1]):
            with dpg.tab_bar(tag="main_tab_bar", callback=self._on_tab_change):
                for tab_def, tab_instance in zip(TAB_DEFINITIONS, self.tabs):
                    tab_tag = f"tab_{tab_def.key}"
                    with dpg.tab(label=tab_def.label, tag=tab_tag) as tab_id:
                        tab_instance.build(parent=tab_id)
        dpg.set_primary_window(self.main_window_tag, True)
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def _apply_theme(self) -> None:
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 10)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 10)
                dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 10)
                dpg.add_theme_color(dpg.mvThemeCol_Button, THEME_COLORS["button"])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, THEME_COLORS["button_hovered"])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, THEME_COLORS["button_active"])
                dpg.add_theme_color(dpg.mvThemeCol_Header, THEME_COLORS["header"])
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, THEME_COLORS["header_hovered"])
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, THEME_COLORS["header_active"])
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, THEME_COLORS["frame_bg"])
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, THEME_COLORS["window_bg"])
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, THEME_COLORS["child_bg"])
            # Dear PyGui draws disabled widgets like enabled ones unless the theme gives them their own colours
            for item_type in (dpg.mvButton, dpg.mvInputInt, dpg.mvInputFloat, dpg.mvInputText, dpg.mvCheckbox,
                              dpg.mvCombo, dpg.mvSliderInt, dpg.mvSliderFloat, dpg.mvDragInt, dpg.mvDragFloat):
                with dpg.theme_component(item_type, enabled_state=False):
                    for col in (dpg.mvThemeCol_Button, dpg.mvThemeCol_ButtonHovered, dpg.mvThemeCol_ButtonActive,
                                dpg.mvThemeCol_FrameBg, dpg.mvThemeCol_FrameBgHovered, dpg.mvThemeCol_FrameBgActive):
                        dpg.add_theme_color(col, THEME_COLORS["disabled_bg"])
                    dpg.add_theme_color(dpg.mvThemeCol_Text, THEME_COLORS["disabled_text"])
        dpg.bind_theme(global_theme)


    def _on_tab_change(self, sender, app_data, user_data):
        try:
            det_tag = "tab_detection"
            det_id = dpg.get_alias_id(det_tag) if dpg.does_item_exist(det_tag) else None
            is_detection = (app_data == det_tag) or (det_id is not None and app_data == det_id)
    
            if is_detection:
                det_tab = self._tabs_by_key.get("detection")
                hook = getattr(det_tab, "on_tab_selected", None)
                if callable(hook):
                    hook()
        except Exception as e:  # never break tab switching, but do not hide the error either
            print(f"[LAYOUT] tab change hook failed: {e}")