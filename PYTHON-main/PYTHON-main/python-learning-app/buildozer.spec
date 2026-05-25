[app]
title = PyLearn
package.name = pylearn
package.domain = com.yourdomain

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json

version = 1.0.0

requirements = python3,kivy==2.3.0,kivymd

orientation = portrait

osx.python_version = 3
osx.kivy_version = 2.3.0

fullscreen = 0

android.minapi = 21
android.ndk = 25b
android.sdk = 34
android.accept_sdk_license = True
android.arch = arm64-v8a

android.permissions = INTERNET,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE

android.logcat_filters = *:S python:D

[buildozer]
log_level = 2
warn_on_root = 1
