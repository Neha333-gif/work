from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.metrics import dp
from kivy.app import App
import sys
import io
import traceback


STARTER_CODE = '''\
# Write your Python code here!
name = "World"
print(f"Hello, {name}!")

# Try a loop:
for i in range(5):
    print(i * i)
'''


def safe_exec(code):
    """Execute code safely and capture output."""
    output_buf = io.StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = output_buf
    sys.stderr = output_buf

    # Restrict dangerous builtins
    safe_globals = {
        "__builtins__": {
            "print": print,
            "range": range,
            "len": len,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "reversed": reversed,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "type": type,
            "isinstance": isinstance,
            "input": lambda _="": "",
        }
    }

    try:
        exec(compile(code, "<editor>", "exec"), safe_globals)
    except Exception:
        traceback.print_exc()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    return output_buf.getvalue()


class EditorScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        root = BoxLayout(orientation="vertical", padding=dp(12), spacing=dp(8))

        # Top bar
        top = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8))
        back_btn = Button(
            text="← Back",
            size_hint=(None, 1),
            width=dp(90),
            background_color=(0.2, 0.2, 0.3, 1),
            background_normal="",
            font_size=dp(13),
        )
        back_btn.bind(on_press=lambda _: setattr(App.get_running_app().root, "current", "home"))
        top.add_widget(back_btn)
        top.add_widget(Label(
            text="Code Editor",
            font_size=dp(20),
            bold=True,
            color=(1, 1, 1, 1),
            halign="left",
        ))
        run_btn = Button(
            text="▶  Run",
            size_hint=(None, 1),
            width=dp(90),
            background_color=(0.15, 0.65, 0.35, 1),
            background_normal="",
            font_size=dp(14),
            bold=True,
        )
        run_btn.bind(on_press=self._run_code)
        top.add_widget(run_btn)
        root.add_widget(top)

        # Code editor
        self.code_input = TextInput(
            text=STARTER_CODE,
            font_name="RobotoMono-Regular",
            font_size=dp(13),
            background_color=(0.1, 0.12, 0.18, 1),
            foreground_color=(0.85, 0.95, 1, 1),
            cursor_color=(0.4, 0.8, 1, 1),
            multiline=True,
            size_hint_y=0.55,
            padding=[dp(10), dp(10)],
        )
        root.add_widget(self.code_input)

        # Output label
        root.add_widget(Label(
            text="Output:",
            font_size=dp(13),
            color=(0.5, 0.5, 0.7, 1),
            size_hint_y=None,
            height=dp(22),
            halign="left",
        ))

        # Output box
        self.output_box = TextInput(
            text="",
            font_name="RobotoMono-Regular",
            font_size=dp(13),
            background_color=(0.06, 0.08, 0.12, 1),
            foreground_color=(0.5, 1, 0.6, 1),
            readonly=True,
            multiline=True,
            size_hint_y=0.35,
            padding=[dp(10), dp(10)],
        )
        root.add_widget(self.output_box)

        # Clear btn
        clear_btn = Button(
            text="Clear output",
            size_hint_y=None,
            height=dp(38),
            background_color=(0.25, 0.15, 0.15, 1),
            background_normal="",
            font_size=dp(13),
        )
        clear_btn.bind(on_press=lambda _: setattr(self.output_box, "text", ""))
        root.add_widget(clear_btn)

        self.add_widget(root)

    def _run_code(self, *args):
        code = self.code_input.text
        result = safe_exec(code)
        self.output_box.text = result if result else "(no output)"
        self.output_box.foreground_color = (
            (1, 0.45, 0.45, 1) if "Error" in result or "Traceback" in result
            else (0.5, 1, 0.6, 1)
        )
