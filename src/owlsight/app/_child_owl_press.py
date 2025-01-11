#!/usr/bin/env python3

import sys
import json
import time
from pynput.keyboard import Key, Controller

def main():
    # Make sure we have the JSON argument
    if len(sys.argv) < 2:
        print("No JSON parameters passed to _child_owl_press.py.")
        return

    # 1) Parse the JSON from sys.argv[1]
    params_json = sys.argv[1]
    try:
        params = json.loads(params_json)
    except json.JSONDecodeError:
        print("Invalid JSON passed to _child_owl_press.py.")
        return

    # 2) Extract the parameters
    sequence = params.get("sequence", [])
    time_before_sequence = float(params.get("time_before_sequence", 0.5))
    time_between_keys = float(params.get("time_between_keys", 0.12))

    time.sleep(time_before_sequence)

    controller = Controller()
    key_map = {
        "L": Key.left,
        "R": Key.right,
        "U": Key.up,
        "D": Key.down,
        "ENTER": Key.enter,
    }

    for char in sequence:
        if char in key_map:
            controller.tap(key_map[char])
        elif char == " ":
            controller.tap(Key.space)
        else:
            controller.type(char)
        time.sleep(time_between_keys)

if __name__ == "__main__":
    main()
