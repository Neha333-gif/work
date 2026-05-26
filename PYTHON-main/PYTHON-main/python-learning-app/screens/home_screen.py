from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.graphics import Color, RoundedRectangle
from kivy.metrics import dp
from kivy.app import App


def _rounded_bg(widget, r, g, b, a=1, radius=12):
    with widget.canvas.before:
        Color(r, g, b, a)
        rect = RoundedRectangle(pos=widget.pos, size=widget.size, radius=[dp(radius)])
    widget.bind(
        pos=lambda *_: setattr(rect, "pos", widget.pos),
        size=lambda *_: setattr(rect, "size", widget.size),
    )


class MenuCard(BoxLayout):
    def __init__(self, title, subtitle, icon, color, screen_name, **kwargs):
        super().__init__(orientation="vertical", padding=dp(14), spacing=dp(6), **kwargs)
        self.screen_name = screen_name
        self.size_hint_y = None
        self.height = dp(100)

        _rounded_bg(self, *color, radius=12)

        # Title row
        title_lbl = Label(
            text=f"{icon}  {title}",
            font_size=dp(17),
            bold=True,
            color=(1, 1, 1, 1),
            halign="left",
            valign="middle",
            size_hint_y=None,
            height=dp(28),
        )
        title_lbl.bind(size=lambda w, _: setattr(w, "text_size", (w.width, None)))
        self.add_widget(title_lbl)

        sub_lbl = Label(
            text=subtitle,
            font_size=dp(12),
            color=(1, 1, 1, 0.78),
            halign="left",
            valign="top",
            size_hint_y=None,
            height=dp(20),
        )
        sub_lbl.bind(size=lambda w, _: setattr(w, "text_size", (w.width, None)))
        self.add_widget(sub_lbl)

        btn_row = BoxLayout(size_hint_y=None, height=dp(28))
        btn = Button(
            text="Open →",
            font_size=dp(12),
            size_hint=(None, 1),
            width=dp(84),
            background_color=(1, 1, 1, 0.22),
            background_normal="",
        )
        btn.bind(on_press=lambda _: self.navigate())
        btn_row.add_widget(btn)
        btn_row.add_widget(Label())  # spacer
        self.add_widget(btn_row)

    def navigate(self):
        App.get_running_app().root.current = self.screen_name


class HomeScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        root = BoxLayout(orientation="vertical", padding=dp(20), spacing=dp(14))

        # Header
        header = BoxLayout(orientation="vertical", size_hint_y=None, height=dp(76), spacing=dp(2))
        header.add_widget(Label(
            text="PyLearn",
            font_size=dp(34),
            bold=True,
            color=(1, 1, 1, 1),
            halign="left",
            size_hint_y=None,
            height=dp(48),
        ))
        sub = Label(
            text="Learn Python — step by step",
            font_size=dp(13),
            color=(0.6, 0.65, 0.85, 1),
            halign="left",
            size_hint_y=None,
            height=dp(22),
        )
        sub.bind(size=lambda w, _: setattr(w, "text_size", (w.width, None)))
        header.add_widget(sub)
        root.add_widget(header)

        scroll = ScrollView(do_scroll_x=False, bar_width=dp(4))
        grid = GridLayout(
            cols=1,
            spacing=dp(10),
            size_hint_y=None,
            padding=[0, 0, 0, dp(20)],
        )
        grid.bind(minimum_height=grid.setter("height"))

        cards = [
            ("Lessons",      "12 topics from basics to advanced",   "📚", (0.22, 0.32, 0.78, 1), "lesson"),
            ("Quizzes",      "Test your knowledge per topic",        "🧠", (0.12, 0.58, 0.50, 1), "quiz"),
            ("Code Editor",  "Write and run Python in-app",         "⌨️",  (0.68, 0.36, 0.14, 1), "editor"),
            ("My Progress",  "Track completed lessons & scores",     "📈", (0.42, 0.18, 0.72, 1), "progress"),
        ]

        for title, subtitle, icon, color, screen in cards:
            grid.add_widget(MenuCard(title, subtitle, icon, color, screen))

        scroll.add_widget(grid)
        root.add_widget(scroll)
        self.add_widget(root)
