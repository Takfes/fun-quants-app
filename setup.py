from setuptools import setup

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().split("\n")

setup(
    name="my-quants-app",
    version="0.0.1",
    install_requires=requirements,
    py_modules=["src"],
)
