from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, SlideTransition
from kivy.core.window import Window
from kivy.utils import platform

from screens.home_screen import HomeScreen
from screens.lesson_screen import LessonScreen
from screens.quiz_screen import QuizScreen
from screens.editor_screen import EditorScreen
from screens.progress_screen import ProgressScreen
from utils.storage import Storage

Window.clearcolor = (0.08, 0.08, 0.12, 1)


class PythonLearningApp(App):
    def build(self):
        self.storage = Storage()
        self.title = "PyLearn"
        self.icon = "assets/icon.png"

        sm = ScreenManager(transition=SlideTransition())
        sm.add_widget(HomeScreen(name="home"))
        sm.add_widget(LessonScreen(name="lesson"))
        sm.add_widget(QuizScreen(name="quiz"))
        sm.add_widget(EditorScreen(name="editor"))
        sm.add_widget(ProgressScreen(name="progress"))
        return sm

    def on_pause(self):
        return True

    def on_resume(self):
        pass


if __name__ == "__main__":
    PythonLearningApp().run()
