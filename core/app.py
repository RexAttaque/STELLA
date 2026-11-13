from __future__ import annotations

import dearpygui.dearpygui as dpg

from config.constants import APP_TITLE
from config.settings import STORE, load_user_settings, save_user_settings, user_settings_path
from interface import PreprocessingTab, DetectionTab, TrackingTab, ValidationTab
from core.layout import LayoutManager


class EVTrackApp:
    def __init__(self) -> None:
        self.preprocessing_tab = PreprocessingTab()
        self.detection_tab = DetectionTab(self.preprocessing_tab)
        self.tracking_tab = TrackingTab(self.preprocessing_tab, self.detection_tab)
        self.validation_tab = ValidationTab(self.preprocessing_tab, self.tracking_tab)
        self.layout = LayoutManager(
            [
                self.preprocessing_tab,
                self.detection_tab,
                self.tracking_tab,
                self.validation_tab,
                
            ]
        )

    def start(self) -> None:
        changed = load_user_settings()  # before the fields are built
        if changed:
            print(f"Settings: {len(changed)} value(s) differ from the defaults ({', '.join(changed)}), "
                  f"from {user_settings_path()} (Settings > Algorithm settings...)")
        self.layout.build()
        STORE.add_listener(lambda key, value: self._save_settings())
        while dpg.is_dearpygui_running():
            self.detection_tab.process_background_tasks()
            self.tracking_tab.process_background_tasks()
            self.preprocessing_tab.timeline.tick()  # slider dragging, playback, bar colours
            self.tracking_tab.sync_pseudoframe_overlay()  # follow the frame shown on the other pages
            self.validation_tab.sync_pseudoframe_overlay()
            dpg.render_dearpygui_frame()
        self._save_settings()
        dpg.destroy_context()

    @staticmethod
    def _save_settings() -> None:
        try:
            save_user_settings()
        except OSError as e:
            print(f"Settings could not be saved: {e}")


def run() -> None:
    print("=" * 60)
    print(APP_TITLE)
    print("=" * 60)
    app = EVTrackApp()
    app.start()

