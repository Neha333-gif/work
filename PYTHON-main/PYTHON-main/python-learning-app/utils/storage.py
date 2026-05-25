import json
import os


class Storage:
    """Simple persistent key-value store backed by a JSON file."""

    def __init__(self, filename="pylearn_data.json"):
        from kivy.app import App
        try:
            data_dir = App.get_running_app().user_data_dir
        except Exception:
            data_dir = "."
        self._path = os.path.join(data_dir, filename)
        self._data = self._load()

    def _load(self):
        try:
            with open(self._path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self):
        try:
            with open(self._path, "w") as f:
                json.dump(self._data, f, indent=2)
        except Exception:
            pass

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value):
        self._data[key] = value
        self._save()

    def delete(self, key):
        self._data.pop(key, None)
        self._save()

    def clear(self):
        self._data = {}
        self._save()
