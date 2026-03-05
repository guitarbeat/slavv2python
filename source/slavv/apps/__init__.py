"""User-facing applications for SLAVV.

Submodules
----------
cli      — Command-line interface  (``slavv`` command)
web_app  — Streamlit web application
"""
from .cli import main   # re-export so entry-point can use slavv.apps:main

__all__ = ["main"]
