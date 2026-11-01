"""The Flask application object; route modules register on it when they are imported (see server.py).
"""
from flask import Flask


app = Flask(__name__)
