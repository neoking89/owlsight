"""
This script is being executed as child process by the owl_press function
"""

from typing import List
import time
import sys
import json
from pynput.keyboard import Controller, Key


KEY_MAP = {
    "L": Key.left,
    "R": Key.right,
    "U": Key.up,
    "D": Key.down,
    "ENTER": Key.enter,
    "DEL": Key.delete,
    "CTRL+A": (Key.ctrl, "a"),
    "CTRL+C": (Key.ctrl, "c"),
    "CTRL+Y": (Key.ctrl, "y"),
}


def execute_key_sequence(sequence: List[str], time_before_sequence: float, time_between_keys: float):
    time.sleep(time_before_sequence)
    controller = Controller()

    for item in sequence:
        if item.startswith("SLEEP:"):
            try:
                time.sleep(float(item.split(":")[1]))
            except Exception as e:
                print(f"Invalid sleep time {item}: ", e)
            continue

        if item in KEY_MAP:
            key = KEY_MAP[item]
            if isinstance(key, tuple):
                # Handle modifier combinations
                modifier, char = key
                with controller.pressed(modifier):
                    controller.tap(char)
            else:
                controller.tap(key)
        else:
            controller.type(item)
        time.sleep(time_between_keys)


def main():
    # Make sure we have the JSON argument
    if len(sys.argv) < 2:
        print("No JSON parameters passed to _child_owl_press.py.")
        return

    # Parse the JSON from sys.argv[1]
    params_json = sys.argv[1]
    try:
        params = json.loads(params_json)
    except json.JSONDecodeError:
        print("Invalid JSON passed to _child_owl_press.py.")
        return

    sequence = params.get("sequence", [])
    time_before_sequence = float(params.get("time_before_sequence", 0.5))
    time_between_keys = float(params.get("time_between_keys", 0.12))

    execute_key_sequence(sequence, time_before_sequence, time_between_keys)


if __name__ == "__main__":
    main()
