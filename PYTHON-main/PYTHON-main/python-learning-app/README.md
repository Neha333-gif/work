# PyLearn — Python Learning App (Kivy)

A mobile Python learning app built with Kivy. Features:

- 📚 **Lessons** — 6+ theory topics with code examples
- 🧠 **Quizzes** — 8 multiple-choice questions with scoring
- ⌨️ **Code Editor** — Write and run Python code in-app (sandboxed)
- 📈 **Progress** — Mark lessons done, track completion %

---

## Project structure

```
python-learning-app/
├── main.py                     # App entry point
├── buildozer.spec              # Android build config
├── requirements.txt
├── screens/
│   ├── home_screen.py
│   ├── lesson_screen.py
│   ├── quiz_screen.py
│   ├── editor_screen.py
│   └── progress_screen.py
├── utils/
│   └── storage.py              # JSON-backed persistence
└── .github/
    └── workflows/
        └── build.yml           # GitHub Actions → APK
```

---

## Run locally

```bash
pip install kivy==2.3.0
python main.py
```

---

## Build Android APK via GitHub Actions

1. Push this repo to GitHub (any branch named `main`)
2. GitHub Actions runs automatically
3. Go to **Actions → Build Android APK → Artifacts**
4. Download `PyLearn-debug-apk`

To trigger a release APK, create a git tag:
```bash
git tag v1.0.0
git push origin v1.0.0
```
This uploads the APK to **GitHub Releases** automatically.

---

## Customise

- **Add lessons** → edit `LESSONS` list in `screens/lesson_screen.py`
- **Add quiz questions** → edit `QUIZ_QUESTIONS` in `screens/quiz_screen.py`
- **Change app name / package** → edit `buildozer.spec`

---

## iOS note

iOS builds require a Mac with Xcode and an Apple Developer account. The GitHub Actions workflow builds Android only. For iOS you'll need to run `toolchain build kivy` locally using `kivy-ios`.
