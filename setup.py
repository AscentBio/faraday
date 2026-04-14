# setup.py

from setuptools import setup, find_packages, find_namespace_packages
from pathlib import Path


def load_requirements():
    req_path = Path(__file__).with_name("requirements.txt")
    if not req_path.exists():
        # Dependencies are declared in pyproject.toml for uv/pip builds.
        # Keep setup.py compatible for legacy editable installs.
        return []
    return [
        line.strip()
        for line in req_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

setup(
    name="faradayai",
    version="0.1.4",
    install_requires=load_requirements(),
    packages=find_namespace_packages(include=['faraday*']),
    entry_points={
        "console_scripts": [
            "faraday=faraday.cli:main",
        ],
    },
)
