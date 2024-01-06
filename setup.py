from setuptools import find_packages, setup

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().split("\n")

if __name__ == "__main__":
    setup(
        name="quanttak",
        version="0.0.1",
        install_requires=requirements,
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        # install_requires=find_packages(),
        py_modules=["src"],
    )
