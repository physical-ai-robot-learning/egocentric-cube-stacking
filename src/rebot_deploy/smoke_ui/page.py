"""The single-page HTML/JS front end, served by GET /. The markup lives in page.html next to this file and is read once,
as bytes, so the served page is byte-identical to the string that used to be inlined in mac_v4_smoke_ui.py.
"""
from pathlib import Path

PAGE = (Path(__file__).with_name("page.html")).read_bytes().decode("utf-8")
