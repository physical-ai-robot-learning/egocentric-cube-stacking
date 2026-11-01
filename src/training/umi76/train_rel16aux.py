"""[2026-09-29] lerobot_train with the REL16-v3 aux plugin installed first (rel16_aux.py). Same CLI as lerobot_train.py."""
import os, sys
try:
    from . import rel16_aux
except ImportError:  # run as a flat script (its folder is sys.path[0])
    import rel16_aux
rel16_aux.install()
from lerobot.scripts.lerobot_train import main
main()
