"""Robot service pass-through routes: /connect, /estop, /clear_estop, /reenable, /rest.
"""
from flask import jsonify

from . import state as rt
from .app import app
from .robot_client import _jaw_torque, _robot


@app.route("/connect", methods=["POST"])
def connect():
    return jsonify(_robot("/connect", "post", json={}))


@app.route("/estop", methods=["POST"])
def estop():
    rt.STATE["busy"] = False
    rt.STATE["running"] = False
    rt.STATE["stop_reason"] = "estop"
    return jsonify(_robot("/estop", "post", json={}))


@app.route("/clear_estop", methods=["POST"])
def clear_estop():
    return jsonify(_robot("/clear_estop", "post", json={}))


@app.route("/reenable", methods=["POST"])
def reenable():
    return jsonify(_robot("/reenable", "post", json={}))


@app.route("/rest", methods=["POST"])
def rest():
    rt.STATE["running"] = False
    rt.STATE["stop_reason"] = "rest"
    r = _robot("/rest", "post", json={})
    jaw = _jaw_torque(False)           # nothing to hold at rest, so nothing to buzz about
    return jsonify({"rest": r, "jaw_torque_off": jaw})
