from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.metrics import dp
from kivy.app import App
from kivy.graphics import Color, RoundedRectangle

QUIZ_QUESTIONS = [
    {
        "question": "What is the output of: print(type(3.14))",
        "options": ["<class 'int'>", "<class 'float'>", "<class 'str'>", "<class 'number'>"],
        "answer": 1,
    },
    {
        "question": "Which keyword defines a function in Python?",
        "options": ["func", "define", "def", "function"],
        "answer": 2,
    },
    {
        "question": "What does len([1, 2, 3]) return?",
        "options": ["2", "3", "4", "Error"],
        "answer": 1,
    },
    {
        "question": "Which of these is a valid dictionary?",
        "options": ['["a", 1]', '("a", 1)', '{"a": 1}', '{a = 1}'],
        "answer": 2,
    },
    {
        "question": "What is the result of 5 // 2 in Python?",
        "options": ["2.5", "3", "2", "1"],
        "answer": 2,
    },
    {
        "question": "How do you start a for loop over a list 'items'?",
        "options": ["for items:", "for i in items:", "foreach item in items:", "loop items:"],
        "answer": 1,
    },
    {
        "question": "Which method adds an item to a list?",
        "options": ["list.add()", "list.push()", "list.insert()", "list.append()"],
        "answer": 3,
    },
    {
        "question": "What does 'break' do inside a loop?",
        "options": [
            "Skips current iteration",
            "Exits the loop entirely",
            "Restarts the loop",
            "Causes a syntax error",
        ],
        "answer": 1,
    },
    {
        "question": "What does the 'pass' statement do?",
        "options": [
            "Ends the program",
            "Does nothing (placeholder)",
            "Skips the next line",
            "Returns None",
        ],
        "answer": 1,
    },
    {
        "question": "Which operator checks if two values are equal?",
        "options": ["=", ":=", "==", "==="],
        "answer": 2,
    },
]


def _rounded_bg(widget, r, g, b, a=1, radius=10):
    with widget.canvas.before:
        Color(r, g, b, a)
        rect = RoundedRectangle(pos=widget.pos, size=widget.size, radius=[dp(radius)])
    widget.bind(
        pos=lambda *_: setattr(rect, "pos", widget.pos),
        size=lambda *_: setattr(rect, "size", widget.size),
    )


class QuizScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_q = 0
        self.score = 0
        self.answered = False
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
        back_btn.bind(on_press=self._go_home)
        top.add_widget(back_btn)

        self.progress_label = Label(
            text=f"Q {self.current_q + 1} / {len(QUIZ_QUESTIONS)}",
            font_size=dp(13),
            color=(0.6, 0.65, 0.85, 1),
            halign="right",
            valign="middle",
        )
        self.progress_label.bind(size=lambda w, _: setattr(w, "text_size", (w.width, None)))
        top.add_widget(self.progress_label)
        root.add_widget(top)

        # ── score bar ────────────────────────────
        self.score_label = Label(
            text=f"Score: {self.score} / {len(QUIZ_QUESTIONS)}",
            font_size=dp(14),
            color=(0.35, 0.95, 0.65, 1),
            size_hint_y=None,
            height=dp(26),
            halign="left",
            valign="middle",
        )
        self.score_label.bind(size=lambda w, _: setattr(w, "text_size", (w.width, None)))
        root.add_widget(self.score_label)

        # ── question card ────────────────────────
        q_card = BoxLayout(
            orientation="vertical",
            padding=[dp(14), dp(10)],
            size_hint_y=None,
        )
        _rounded_bg(q_card, 0.13, 0.17, 0.28, radius=12)

        q_text = QUIZ_QUESTIONS[self.current_q]["question"]
        self.q_label = Label(
            text=q_text,
            font_size=dp(14),
            color=(1, 1, 1, 1),
            halign="left",
            valign="top",
            size_hint_y=None,
        )
        self.q_label.bind(
            width=lambda w, _: setattr(w, "text_size", (w.width, None)),
            texture_size=lambda w, _: setattr(w, "height", w.texture_size[1]),
        )
        q_card.bind(minimum_height=lambda w, h: setattr(w, "height", h + dp(20)))
        q_card.add_widget(self.q_label)
        root.add_widget(q_card)

        # ── answer options ────────────────────────
        self.option_buttons = []
        options = QUIZ_QUESTIONS[self.current_q]["options"]
        for i, opt in enumerate(options):
            btn = Button(
                text=opt,
                font_size=dp(13),
                size_hint_y=None,
                height=dp(50),
                background_color=(0.16, 0.18, 0.30, 1),
                background_normal="",
                color=(1, 1, 1, 1),
                halign="left",
                padding_x=dp(12),
            )
            btn.bind(on_press=lambda b, idx=i: self._answer(idx))
            self.option_buttons.append(btn)
            root.add_widget(btn)

        # ── feedback label ────────────────────────
        self.feedback_label = Label(
            text="",
            font_size=dp(13),
            size_hint_y=None,
            height=dp(32),
            color=(1, 1, 1, 1),
            halign="left",
            valign="middle",
        )
        self.feedback_label.bind(size=lambda w, _: setattr(w, "text_size", (w.width, None)))
        root.add_widget(self.feedback_label)

        # ── next button ───────────────────────────
        self.next_btn = Button(
            text="Next →",
            size_hint_y=None,
            height=dp(48),
            background_color=(0.22, 0.42, 0.82, 1),
            background_normal="",
            font_size=dp(14),
            bold=True,
            opacity=0,
            disabled=True,
        )
        self.next_btn.bind(on_press=self._next_question)
        root.add_widget(self.next_btn)

        root.add_widget(Label())  # spacer
        self.add_widget(root)

    def _answer(self, idx):
        if self.answered:
            return
        self.answered = True
        correct = QUIZ_QUESTIONS[self.current_q]["answer"]

        for i, btn in enumerate(self.option_buttons):
            btn.disabled = True
            if i == correct:
                btn.background_color = (0.10, 0.72, 0.38, 1)
            elif i == idx:
                btn.background_color = (0.78, 0.18, 0.18, 1)

        if idx == correct:
            self.score += 1
            self.feedback_label.text = "✓  Correct!"
            self.feedback_label.color = (0.25, 1, 0.55, 1)
        else:
            correct_text = QUIZ_QUESTIONS[self.current_q]["options"][correct]
            self.feedback_label.text = f"✗  Correct: {correct_text}"
            self.feedback_label.color = (1, 0.38, 0.38, 1)

        self.score_label.text = f"Score: {self.score} / {len(QUIZ_QUESTIONS)}"
        self.next_btn.opacity = 1
        self.next_btn.disabled = False

    def _next_question(self, *args):
        self.answered = False
        self.current_q += 1
        if self.current_q >= len(QUIZ_QUESTIONS):
            self._show_results()
        else:
            self._build_ui()

    def _show_results(self):
        self.clear_widgets()
        layout = BoxLayout(orientation="vertical", padding=dp(30), spacing=dp(14))
        pct = int(self.score / len(QUIZ_QUESTIONS) * 100)
        emoji = "🏆" if pct >= 80 else "👍" if pct >= 50 else "📚"

        layout.add_widget(Label(text=emoji, font_size=dp(52), size_hint_y=None, height=dp(72)))
        layout.add_widget(Label(
            text="Quiz Complete!",
            font_size=dp(26),
            bold=True,
            color=(1, 1, 1, 1),
            size_hint_y=None,
            height=dp(40),
        ))
        layout.add_widget(Label(
            text=f"{self.score} / {len(QUIZ_QUESTIONS)} correct ({pct}%)",
            font_size=dp(17),
            color=(0.35, 0.95, 0.65, 1),
            size_hint_y=None,
            height=dp(32),
        ))

        retry_btn = Button(
            text="Retry Quiz",
            size_hint_y=None, height=dp(52),
            background_color=(0.22, 0.42, 0.82, 1),
            background_normal="",
            font_size=dp(15),
        )
        retry_btn.bind(on_press=self._restart)
        layout.add_widget(retry_btn)

        home_btn = Button(
            text="← Home",
            size_hint_y=None, height=dp(48),
            background_color=(0.18, 0.20, 0.30, 1),
            background_normal="",
            font_size=dp(14),
        )
        home_btn.bind(on_press=self._go_home)
        layout.add_widget(home_btn)

        layout.add_widget(Label())  # spacer
        self.add_widget(layout)

    def _restart(self, *args):
        self.current_q = 0
        self.score = 0
        self.answered = False
        self._build_ui()

    def _go_home(self, *args):
        App.get_running_app().root.current = "home"
