import os
from pathlib import Path

from setuptools import setup, find_packages

if __name__ == "__main__":
    with Path(Path(__file__).parent, "README.md").open(encoding="utf-8") as file:
        long_description = file.read()

    def _read_reqs(relpath):
        fullpath = os.path.join(os.path.dirname(__file__), relpath)
        with open(fullpath) as f:
            return [
                s.strip()
                for s in f.readlines()
                if (s.strip() and not s.startswith("#"))
            ]

    REQUIREMENTS = _read_reqs("requirements.txt")

    setup(
        name="holodeck",
        packages=find_packages(),
        include_package_data=True,
        version="0.0.2",
        license="Apache 2.0",
        description='Holodeck: a framework for "Language Guided Generation of 3D Embodied AI Environments".',
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Allen Institute for AI",
        author_email="lucaw@allenai.org",
        url="https://github.com/allenai/Holodeck",
        data_files=[(".", ["README.md"])],
        keywords=[
            "procedural generation",
            "home environments",
            "unity",
            "3D assets",
            "annotation",
            "3D",
            "ai2thor",
        ],
        install_requires=REQUIREMENTS,
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
        package_data={
            "objathor": ["generation/*/*.json", "generation/*.json"],
        },
    )
