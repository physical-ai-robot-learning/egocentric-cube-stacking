"""[2026-08-30] Map Damiao arm controllers (USB CDC; both boards report serial 00000000050C, so /dev names follow plug order)
to their physical USB port (locationID) and tty path, using the IOSerialBSDClient ancestry tree (macOS `ioreg`; deploy stage).

In:  the live USB registry.  Out: {locationID (int): "/dev/cu.usbmodemXXXX"}; the robot service picks the arms by LEFT_LOC /
RIGHT_LOC from it.
usage: python -m rebot_deploy.robot_service.usb_arm_ports   |   from rebot_deploy.robot_service.usb_arm_ports import armPorts
"""
from __future__ import annotations

import re
import subprocess
import sys

IOREG_CMD = ["ioreg", "-r", "-c", "IOSerialBSDClient", "-t", "-l", "-w0"]
DEVICE_MARK = "+-o CDC Device@"
LOCATION_RE = re.compile(r'"locationID" = (\d+)')
CALLOUT_RE = re.compile(r'"IOCalloutDevice" = "([^"]+)"')


def armPorts() -> dict[int, str]:
    """{locationID: callout tty} of every CDC device in the registry (was arm_ports)"""
    tText = subprocess.run(IOREG_CMD, capture_output=True, text=True).stdout
    tPorts, tLoc, tInDevice = {}, None, False
    for tLine in tText.splitlines():
        if DEVICE_MARK in tLine:
            tLoc, tInDevice = None, True
            continue
        if tInDevice and tLoc is None:
            tMatch = LOCATION_RE.search(tLine)
            if tMatch:
                tLoc = int(tMatch.group(1))
                continue
        if tInDevice and tLoc is not None:
            tMatch = CALLOUT_RE.search(tLine)
            if tMatch:
                tPorts[tLoc] = tMatch.group(1)
                tLoc, tInDevice = None, False
    return tPorts


def main(argv: list[str] | None = None) -> int:
    tPorts = armPorts()
    for tLoc, tPath in sorted(tPorts.items()):
        print(f"loc={tLoc} (0x{tLoc:08x})  {tPath}")
    if not tPorts:
        print("no arm controllers found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
