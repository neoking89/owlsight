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
    " ": Key.space,
}


def execute_key_sequence(sequence: List[str], time_before_sequence: float, time_between_keys: float):
    """
    Executes a sequence of keys and/or functions with a delay between each key press.

    Parameters:
    ----------
    sequence: List[str, callable]
        A list of keys and/or functions to execute.
    time_before_sequence: float
        The time to wait before executing the sequence.
    time_between_keys: float
        The time to wait between each key press in the sequence

    Returns:
    -------
    None
    """
    time.sleep(time_before_sequence)

    controller = Controller()

    for item in sequence:
        if item in KEY_MAP:
            controller.tap(KEY_MAP[item])
        elif item.startswith("SLEEP:"):
            try:
                sleep_time = float(item.split(":")[1])
                time.sleep(sleep_time)
            except (ValueError, IndexError):
                print(f"Invalid sleep time: {item}. Skipping item.")
                continue
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
