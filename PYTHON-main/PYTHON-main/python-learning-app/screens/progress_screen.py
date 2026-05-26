from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.progressbar import ProgressBar
from kivy.metrics import dp
from kivy.app import App
from kivy.graphics import Color, RoundedRectangle

TOPICS = [
    "Variables & Data Types",
    "Strings",
    "Control Flow",
    "Loops",
    "Functions",
    "Lists & Tuples",
    "Dictionaries & Sets",
    "Error Handling",
    "Object-Oriented Programming",
    "File I/O",
    "Modules & Packages",
    "Comprehensions & Generators",
]


def _rounded_bg(widget, r, g, b, a=1, radius=8):
    with widget.canvas.before:
        Color(r, g, b, a)
        rect = RoundedRectangle(pos=widget.pos, size=widget.size, radius=[dp(radius)])
    widget.bind(
        pos=lambda *_: setattr(rect, "pos", widget.pos),
        size=lambda *_: setattr(rect, "size", widget.size),
    )


class ProgressScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._build_ui()

    def _build_ui(self):
        self.clear_widgets()
        root = BoxLayout(orientation="vertical", padding=dp(14), spacing=dp(10))

        # ── top bar ──────────────────────────────
        top = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8))
        back_btn = Button(
            text="← Back",
            size_hint=(None, 1),
            width=dp(90),
            background_color=(0.18, 0.20, 0.30, 1),
            background_normal="",
            font_size=dp(13),
        )
        back_btn.bind(on_press=lambda _: setattr(App.get_running_app().root, "current", "home"))
        top.add_widget(back_btn)
        top.add_widget(Label(
            text="📈  My Progress",
            font_size=dp(20),
            bold=True,
            color=(1, 1, 1, 1),
            halign="left",
        ))
        root.add_widget(top)

        storage = App.get_running_app().storage
        completed = storage.get("completed_lessons", [])

        total = len(TOPICS)
        done  = len(completed)
        pct   = int(done / total * 100) if total else 0

        # ── summary card ─────────────────────────
        summary = BoxLayout(
            orientation="vertical",
            padding=[dp(14), dp(10)],
            spacing=dp(6),
            size_hint_y=None,
            height=dp(88),
        )
        _rounded_bg(summary, 0.14, 0.20, 0.36, radius=10)

        done_lbl = Label(
            text=f"{done} / {total} topics completed",
            font_size=dp(15),
            bold=True,
            color=(1, 1, 1, 1),
            halign="left",
            valign="middle",
            size_hint_y=None,
            height=dp(24),
        )
        done_lbl.bind(size=lambda w, _: setattr(w, "text_size", (w.width, None)))
        summary.add_widget(done_lbl)

        summary.add_widget(ProgressBar(max=100, value=pct, size_hint_y=None, height=dp(12)))

        pct_lbl = Label(
            text=f"{pct}% complete",
            font_size=dp(12),
            color=(0.45, 0.88, 0.68, 1),
            halign="left",
            valign="middle",
            size_hint_y=None,
            height=dp(20),
        )
        pct_lbl.bind(size=lambda w, _: setattr(w, "text_size", (w.width, None)))
        summary.add_widget(pct_lbl)
        root.add_widget(summary)

        # ── topic rows ────────────────────────────
        scroll = ScrollView(do_scroll_x=False, bar_width=dp(4))
        grid = BoxLayout(
            orientation="vertical",
            spacing=dp(6),
            size_hint_y=None,
            padding=[0, 0, 0, dp(16)],
        )
        grid.bind(minimum_height=grid.setter("height"))

        for topic in TOPICS:
            row = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8), padding=[dp(8), 0])
            _rounded_bg(row, 0.11, 0.13, 0.20, radius=8)

            is_done = topic in completed
            status  = "✓" if is_done else "○"
            s_color = (0.25, 1, 0.48, 1) if is_done else (0.44, 0.44, 0.58, 1)

            row.add_widget(Label(
                text=status,
                font_size=dp(16),
                color=s_color,
                size_hint=(None, 1),
                width=dp(28),
            ))

            t_lbl = Label(
                text=topic,
                font_size=dp(13),
                color=(1, 1, 1, 0.92) if is_done else (0.68, 0.68, 0.78, 1),
                halign="left",
                valign="middle",
            )
            t_lbl.bind(size=lambda w, _: setattr(w, "text_size", (w.width, None)))
            row.add_widget(t_lbl)

            toggle = Button(
                text="Done ✓" if not is_done else "Undo",
                size_hint=(None, None),
                size=(dp(72), dp(32)),
                background_color=(0.18, 0.46, 0.78, 1) if not is_done else (0.28, 0.28, 0.38, 1),
                background_normal="",
                font_size=dp(12),
            )
            toggle.bind(on_press=lambda _, t=topic: self._toggle(t))
            row.add_widget(toggle)
            grid.add_widget(row)

        scroll.add_widget(grid)
        root.add_widget(scroll)
        self.add_widget(root)

    def _toggle(self, topic):
        storage = App.get_running_app().storage
        completed = storage.get("completed_lessons", [])
        if topic in completed:
            completed.remove(topic)
        else:
            completed.append(topic)
        storage.set("completed_lessons", completed)
        self._build_ui()
