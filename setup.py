from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autoworkouttracker", # Replace with your own username
    version="0.0.1",
    author="bryce kroencke",
    author_email="brycekroencke@gmail.com",
    description="A package for classifiying and tracking user workouts using ML and computer vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brycekroencke/workout_tracker",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
