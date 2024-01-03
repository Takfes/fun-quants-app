from setuptools import setup

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().split("\n")

setup(
    name="my-quants-app",
    version="0.0.2",
    install_requires=requirements,
    # install_requires=find_packages(),
    py_modules=["src", "structural", "models"],
)
