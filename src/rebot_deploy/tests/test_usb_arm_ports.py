"""Characterization of the USB arm-port mapping (ioreg output parsed; ioreg itself is replaced by a fixed text)."""
from __future__ import annotations

import subprocess
from types import SimpleNamespace

from rebot_deploy.robot_service import usb_arm_ports

IOREG_TEXT = """\
+-o Root  <class IORegistryEntry>
  +-o CDC Device@14100000  <class IOUSBHostDevice>
    |   "locationID" = 336592896
    |   "USB Serial Number" = "00000000050C"
    | +-o IOSerialBSDClient  <class IOSerialBSDClient>
    |     "IOCalloutDevice" = "/dev/cu.usbmodem14101"
  +-o CDC Device@14200000  <class IOUSBHostDevice>
    |   "locationID" = 337641472
    | +-o IOSerialBSDClient
    |     "IOCalloutDevice" = "/dev/cu.usbmodem14201"
    |     "IOCalloutDevice" = "/dev/cu.ignored-second"
  +-o Other Device@1  <class IOUSBHostDevice>
    |   "locationID" = 1
    |     "IOCalloutDevice" = "/dev/cu.not-an-arm"
"""


def _fakeRun(text, calls):
    def run(cmd, capture_output, text_mode=None, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout=text)
    return lambda cmd, capture_output=True, text=True: run(cmd, capture_output)


def test_arm_ports_maps_location_ids_to_callout_devices(monkeypatch):
    tCalls = []
    monkeypatch.setattr(subprocess, "run", _fakeRun(IOREG_TEXT, tCalls))
    assert usb_arm_ports.arm_ports() == {336592896: "/dev/cu.usbmodem14101", 337641472: "/dev/cu.usbmodem14201"}
    assert tCalls == [["ioreg", "-r", "-c", "IOSerialBSDClient", "-t", "-l", "-w0"]]


def test_arm_ports_empty_output(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fakeRun("", []))
    assert usb_arm_ports.arm_ports() == {}


def _runAsScript():
    import runpy
    try:
        runpy.run_module("rebot_deploy.robot_service.usb_arm_ports", run_name="__main__")
    except SystemExit as tExit:
        assert tExit.code in (None, 0)


def test_script_prints_one_line_per_port(monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "run", _fakeRun(IOREG_TEXT, []))
    _runAsScript()
    assert capsys.readouterr().out == ("loc=336592896 (0x14100000)  /dev/cu.usbmodem14101\n"
                                       "loc=337641472 (0x14200000)  /dev/cu.usbmodem14201\n")
    monkeypatch.setattr(subprocess, "run", _fakeRun("", []))
    _runAsScript()
    assert capsys.readouterr().out == "no arm controllers found\n"
