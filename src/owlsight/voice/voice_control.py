import re
import threading
import queue
import time
from typing import Dict, Optional, Callable, Union, List
import logging

from owlsight.utils.helper_functions import validate_input_params
from owlsight.utils.logger import logger

# Flag to track if voice control dependencies are available
VOICE_CONTROL_AVAILABLE = False

try:
    from RealtimeSTT import AudioToTextRecorder
    from pynput.keyboard import Controller, Key

    VOICE_CONTROL_AVAILABLE = True
except ImportError:
    logger.debug(
        "Voice control dependencies not found. Install with 'pip install owlsight[voice]' to enable voice control."
    )


class VoiceControlBase:
    """Base class for voice control functionality."""

    def start(self):
        """Start the voice control system."""
        pass

    def stop(self):
        """Stop the voice control system."""
        pass

    def is_running(self):
        """Check if the voice control system is running."""
        return False


class DummyVoiceControl(VoiceControlBase):
    """Dummy implementation when voice control dependencies are not available."""

    def __init__(self, *args, **kwargs):
        logger.warning(
            "Voice control dependencies not found. Voice control will be disabled. "
            "To enable voice control, install the required dependencies with: "
            "'pip install owlsight[voice]'"
        )


if VOICE_CONTROL_AVAILABLE:

    class VoiceControl(VoiceControlBase):
        def __init__(
            self,
            word_to_key_map: Dict[str, Union[str, List[str]]] = None,
            word_to_word_map: Dict[str, str] = None,
            cmd_cooldown: float = 1.0,
            debug: bool = False,
            language: str = "en",
            model: str = "small.en",
            key_press_interval: float = 0.05,
            typing_interval: float = 0.03,
            on_command_processed: Optional[Callable[[str, Union[str, List[str]]], None]] = None,
            **recorder_kwargs,
        ):
            """
            Initialize the VoiceControl instance.

            Parameters
            ----------
            word_to_key_map : Dict[str, Union[str, List[str]]], optional
                A dictionary mapping words to keys/commands, by default None
                This is used to map words to key/command combinations
                eg: {"zap": ["ctrl", "a"]} means that the word "zap" will be transcribed (and executed) as "ctrl+a"
            word_to_word_map : Dict[str, str], optional
                A dictionary mapping and transcribing words to other words, by default None
                eg: {"exit": "exit()"} means that the word "exit" will be transcribed as "exit()"
            cmd_cooldown : float, optional
                The cooldown between commands, by default 1.0.
                This is to prevent multiple commands being sent too quickly in a sequence.
            debug : bool, optional
                Whether to enable debug mode, by default False
            language : str, optional
                The language to use, by default "en"
            model : str, optional
                The model to use, by default "small.en"
            key_press_interval : float, optional
                The interval between key presses, by default 0.05
            typing_interval : float, optional
                The interval between typing, by default 0.03
            on_command_processed : Optional[Callable[[str, Union[str, List[str]]], None]], optional
                A function to call when a command is processed, by default None
            **recorder_kwargs
                Keyword arguments to pass to the `AudioToTextRecorder` constructor
            """
            super().__init__()
            self.word_to_key_map = {k.lower(): v for k, v in (word_to_key_map or {}).items()}
            self.word_to_word_map = word_to_word_map or {}
            self.cmd_cooldown = cmd_cooldown
            self.debug = debug
            self.language = language
            self.model = model
            self.key_press_interval = key_press_interval
            self.typing_interval = typing_interval
            self.on_command_processed = on_command_processed

            self.keyboard = Controller()

            self.key_press_queue = queue.Queue()
            self.typing_queue = queue.Queue()
            self.recent_commands = {}

            if self.debug:
                logger.setLevel(logging.DEBUG)

            # Store complete commands and their word sets separately
            self.complete_commands = set(self.word_to_key_map.keys())
            self.command_words = set()
            for cmd in self.complete_commands:
                # Only add single-word commands to command_words
                if " " not in cmd:
                    self.command_words.add(cmd.lower())

            # For word transformations
            self.word_transform_patterns = []
            for word, replacement in self.word_to_word_map.items():
                # Pattern matches the word with optional punctuation
                pattern = re.compile(rf"\b{re.escape(word)}\b[.!,;?]*", re.IGNORECASE)
                self.word_transform_patterns.append((pattern, replacement))

            # For sentence end punctuation
            self.sentence_end_punct_pattern = re.compile(r"[.!?]+")

            self.key_press_thread = threading.Thread(target=self._key_press_worker, daemon=True)
            self.typing_thread = threading.Thread(target=self._typing_worker, daemon=True)
            self.voice_thread = None

            self._stop_event = threading.Event()
            self.recorder = None
            self._initialize_recorder(**recorder_kwargs)

        def _initialize_recorder(self, **recorder_kwargs) -> None:
            """Initialize the speech recognition recorder."""
            validate_input_params(AudioToTextRecorder.__init__, recorder_kwargs)

            self.recorder = AudioToTextRecorder(
                language=self.language,
                model=self.model,
                enable_realtime_transcription=True,
                on_realtime_transcription_update=self._trigger_keys,
                spinner=False,
                **recorder_kwargs,
            )

            if self.debug:
                logger.debug("Speech recognition initialized with commands:")
                for word, key in self.word_to_key_map.items():
                    logger.debug(f"'{word}' -> {key}")

        def _transform_text(self, text: str) -> str:
            """Apply word transformations to the text."""
            transformed_text = text
            for pattern, replacement in self.word_transform_patterns:
                transformed_text = pattern.sub(replacement, transformed_text)
            return transformed_text

        def _trigger_keys(self, text: str) -> None:
            """Process real-time transcription updates for key commands."""
            if self.debug:
                logger.debug(f"Real-time update received: {text}")

            self._clean_recent_commands()

            # First apply word transformations
            transformed_text = self._transform_text(text).lower()

            # Remove only .!? before splitting into words
            cleaned_text = self.sentence_end_punct_pattern.sub("", transformed_text)

            words = cleaned_text.split()

            i = 0
            while i < len(words):
                found_command = False
                # Try to match multi-word commands (up to 4 words)
                for cmd_len in range(min(4, len(words) - i), 1, -1):
                    potential_cmd = " ".join(words[i : i + cmd_len])
                    if potential_cmd in self.complete_commands and self._can_process_cmd(potential_cmd):
                        key_combo = self.word_to_key_map[potential_cmd]
                        self.key_press_queue.put(key_combo)
                        if self.on_command_processed:
                            self.on_command_processed(potential_cmd, key_combo)
                        i += cmd_len
                        found_command = True
                        break

                # Then try to match single-word commands
                if not found_command:
                    word = words[i]
                    if word in self.command_words and self._can_process_cmd(word):
                        key_combo = self.word_to_key_map[word]
                        self.key_press_queue.put(key_combo)
                        if self.on_command_processed:
                            self.on_command_processed(word, key_combo)
                    i += 1

        def _process_text(self, text: str) -> None:
            """Process the final transcription for typing text."""
            if self.debug:
                logger.debug(f"Final transcription received: {text}")

            # First apply word transformations
            transformed_text = self._transform_text(text)

            # Check for commands in the transformed text (case-insensitive)
            contains_command = False
            for cmd in self.complete_commands:
                if cmd in transformed_text.lower():
                    contains_command = True
                    break

            # Only process text if it doesn't contain any commands
            if not contains_command and transformed_text.strip():
                if self.debug:
                    logger.debug(f"Queueing transformed text: {transformed_text}")
                self.typing_queue.put(transformed_text)

        def _can_process_cmd(self, cmd: str) -> bool:
            """Check if enough time has passed to process the command again."""
            current_time = time.time()
            if cmd in self.recent_commands and (current_time - self.recent_commands[cmd] < self.cmd_cooldown):
                return False

            # Only add to cooldown if it's a complete command
            if cmd.lower() in self.complete_commands:
                self.recent_commands[cmd] = current_time
                return True

            return True

        def _clean_recent_commands(self) -> None:
            """Remove commands whose cooldown period has expired."""
            current_time = time.time()
            self.recent_commands = {
                word: timestamp
                for word, timestamp in self.recent_commands.items()
                if current_time - timestamp <= self.cmd_cooldown
            }

        def _key_press_worker(self) -> None:
            """Worker thread for processing key presses."""
            if self.debug:
                logger.debug("Key press worker started")
            while True:
                key_combo = self.key_press_queue.get()
                if key_combo is None:
                    self.key_press_queue.task_done()
                    break

                try:
                    if isinstance(key_combo, (list, tuple)):
                        # Press all keys in sequence
                        keys = [getattr(Key, k.lower(), k) if k.lower() in dir(Key) else k for k in key_combo]
                        for key in keys:
                            self.keyboard.press(key)
                            time.sleep(self.key_press_interval)
                        # Release in reverse order
                        for key in reversed(keys):
                            self.keyboard.release(key)
                            time.sleep(self.key_press_interval)
                        if self.debug:
                            logger.debug(f"Pressed key combination: {'+'.join(key_combo)}")
                    else:
                        key = getattr(Key, key_combo.lower(), key_combo) if key_combo.lower() in dir(Key) else key_combo
                        self.keyboard.press(key)
                        time.sleep(self.key_press_interval)
                        self.keyboard.release(key)
                        if self.debug:
                            logger.debug(f"Pressed key: {key_combo}")
                except Exception as e:
                    logger.error(f"Error pressing keys: {e}")
                self.key_press_queue.task_done()

        def _typing_worker(self) -> None:
            """Worker thread for processing text typing."""
            if self.debug:
                logger.debug("Typing worker started")
            while True:
                text = self.typing_queue.get()
                if text is None:
                    self.typing_queue.task_done()
                    break
                try:
                    for char in text:
                        self.keyboard.type(char)
                        time.sleep(self.typing_interval)
                    if self.debug:
                        logger.debug(f"Typed text: {text}")
                except Exception as e:
                    logger.error(f"Error typing text: {e}")
                self.typing_queue.task_done()

        def _voice_worker(self) -> None:
            """Background thread for voice recognition."""
            if self.debug:
                logger.debug("Voice worker started")
            while not self._stop_event.is_set():
                try:
                    self.recorder.text(on_transcription_finished=self._process_text)
                except Exception as e:
                    if self.debug:
                        logger.debug(f"Voice worker encountered exception: {e}")
                    break

        def start(self) -> None:
            """Start the voice control system."""
            self.key_press_thread.start()
            self.typing_thread.start()

            logger.info("Voice control system started")
            logger.info("Available voice commands:")
            for word, key in self.word_to_key_map.items():
                logger.info(f"  Say '{word}' to press '{key}'")

            self.voice_thread = threading.Thread(target=self._voice_worker, daemon=True)
            self.voice_thread.start()

        def stop(self) -> None:
            """Stop the voice control system and clean up resources."""
            if self.debug:
                logger.debug("Stopping voice control system...")

            self._stop_event.set()

            if self.recorder:
                self.recorder.shutdown()

            self.key_press_queue.put(None)
            self.typing_queue.put(None)

            if self.key_press_thread.is_alive():
                self.key_press_thread.join(timeout=5)
            if self.typing_thread.is_alive():
                self.typing_thread.join(timeout=5)
            if self.voice_thread and self.voice_thread.is_alive():
                self.voice_thread.join(timeout=5)

            logger.info("Voice control system stopped")

        @property
        def is_running(self) -> bool:
            """Check if the voice control system is currently running."""
            return (
                self.key_press_thread.is_alive()
                and self.typing_thread.is_alive()
                and self.voice_thread
                and self.voice_thread.is_alive()
                and self.recorder is not None
            )

else:

    class VoiceControl(DummyVoiceControl):
        """Proxy class that inherits from DummyVoiceControl when dependencies are missing"""

        pass


# Example usage (if run directly):
# if __name__ == "__main__":
#     WORD_TO_KEY_MAP = {
#         "left": "left",
#         "right": "right",
#         "up": "up",
#         "down": "down",
#         "enter": "enter",
#         "select all": ["ctrl", "a"],
#         "copy": ["ctrl", "c"],
#         "paste": ["ctrl", "v"],
#         "exit()": "delete",
#     }
#     WORD_TO_WORD_MAP = {
#         "exit": "exit()",
#     }

#     # Optional callback function
#     def on_command(word, key):
#         logger.info(f"Command processed: {word} -> {key}")

#     # Create and start voice control
#     if VOICE_CONTROL_AVAILABLE:
#         vc = VoiceControl(
#             word_to_key_map=WORD_TO_KEY_MAP,
#             word_to_word_map=WORD_TO_WORD_MAP,
#             debug=True,
#             silero_use_onnx=True,
#             on_command_processed=on_command,
#         )
#     else:
#         vc = DummyVoiceControl()

#     try:
#         vc.start()
#         # Keep the main thread alive until a KeyboardInterrupt is received.
#         while True:
#             time.sleep(0.1)
#     except KeyboardInterrupt:
#         vc.stop()
