from RealtimeSTT import AudioToTextRecorder
import pyautogui
import re
import threading
import queue
import time
from typing import Dict, Optional, Callable, Union, List

import sys

sys.path.append("src")
from owlsight.utils.logger import logger


class VoiceControl:
    """
    A class to handle voice-controlled keyboard input with real-time speech recognition.
    Supports command words that trigger key combinations and word transformations.
    """

    def __init__(
        self,
        word_to_key_map: Dict[str, Union[str, List[str]]] = None,
        word_to_word_map: Dict[str, str] = None,
        word_cooldown: float = 0.9,
        debug: bool = False,
        language: str = "en",
        model: str = "small.en",
        key_press_interval: float = 0.05,
        typing_interval: float = 0.03,
        on_command_processed: Optional[Callable[[str, Union[str, List[str]]], None]] = None,
    ):
        """
        Initialize the VoiceControl instance.

        Parameters:
        ----------
            word_to_key_map: Dictionary mapping command words to keyboard keys or key combinations
                Examples:
                    {"up": "up"}  # Single key
                    {"save": ["ctrl", "s"]}  # Key combination
                    {"select all": ["ctrl", "a"]}  # Multiple words to key combination
            word_to_word_map: Dictionary mapping words to their replacements
                Examples:
                    {"exit": "exit()"}  # Replace "exit" with "exit()"
                    {"print": "print()"}  # Replace "print" with "print()"
            word_cooldown: Cooldown period (in seconds) between same command recognition
            debug: Enable debug printing
            language: Language for speech recognition
            model: Model to use for speech recognition
            key_press_interval: Interval between key presses
            typing_interval: Interval between typing characters
            on_command_processed: Optional callback when a command is processed
        """
        # Store commands in lowercase for case-insensitive matching
        self.word_to_key_map = {k.lower(): v for k, v in (word_to_key_map or {}).items()}
        self.word_to_word_map = word_to_word_map or {}
        self.word_cooldown = word_cooldown
        self.debug = debug
        self.language = language
        self.model = model
        self.key_press_interval = key_press_interval
        self.typing_interval = typing_interval
        self.on_command_processed = on_command_processed

        # Initialize PyAutoGUI settings
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = self.key_press_interval

        # Set up queues and tracking
        self.key_press_queue = queue.Queue()
        self.typing_queue = queue.Queue()
        self.recent_words = {}

        # Create sets of words to filter (both key commands and word transformations)
        self.key_command_words = set()
        for cmd in self.word_to_key_map.keys():
            self.key_command_words.update(cmd.lower().split())

        # Compile regex patterns for word matching
        self.non_alpha_pattern = re.compile(r"[^a-zA-Z\s]")

        # Create word transformation patterns
        self.word_transform_patterns = {
            re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE): replacement
            for word, replacement in self.word_to_word_map.items()
        }

        # Initialize worker threads
        self.key_press_thread = threading.Thread(target=self._key_press_worker, daemon=True)
        self.typing_thread = threading.Thread(target=self._typing_worker, daemon=True)
        self.voice_thread = None

        # Initialize recorder
        self.recorder = None
        self._initialize_recorder()

    def _initialize_recorder(self) -> None:
        """Initialize the speech recognition recorder."""
        self.recorder = AudioToTextRecorder(
            language=self.language,
            model=self.model,
            enable_realtime_transcription=True,
            on_realtime_transcription_update=self._trigger_keys,
            spinner=False,
            silero_use_onnx=True,
        )

        if self.debug:
            logger.debug("Speech recognition initialized with commands:")
            for word, key in self.word_to_key_map.items():
                logger.debug(f"'{word}' -> {key}")

    def _key_press_worker(self) -> None:
        """Worker thread for processing key presses."""
        if self.debug:
            logger.debug("Key press worker started")
        while True:
            key_combo = self.key_press_queue.get()
            if key_combo is None:
                break
            if isinstance(key_combo, (list, tuple)):
                # Handle key combination (e.g., ["ctrl", "s"])
                pyautogui.hotkey(*key_combo, interval=self.key_press_interval)
                if self.debug:
                    logger.debug(f"Pressed key combination: {'+'.join(key_combo)}")
            else:
                # Handle single key
                pyautogui.press(key_combo, interval=self.key_press_interval)
                if self.debug:
                    logger.debug(f"Pressed key: {key_combo}")
            self.key_press_queue.task_done()

    def _typing_worker(self) -> None:
        """Worker thread for processing text typing."""
        if self.debug:
            logger.debug("Typing worker started")
        while True:
            text = self.typing_queue.get()
            if text is None:
                break
            pyautogui.write(text, interval=self.typing_interval)
            if self.debug:
                logger.debug(f"Typed text: {text}")
            self.typing_queue.task_done()

    def _can_process_word(self, word: str) -> bool:
        """Check if enough time has passed to process the same word again."""
        current_time = time.time()
        if word in self.recent_words and (current_time - self.recent_words[word] < self.word_cooldown):
            return False
        self.recent_words[word] = current_time
        return True

    def _clean_recent_words(self) -> None:
        """Remove words whose cooldown period has expired."""
        current_time = time.time()
        self.recent_words = {
            word: timestamp
            for word, timestamp in self.recent_words.items()
            if current_time - timestamp <= self.word_cooldown
        }

    def _trigger_keys(self, text: str) -> None:
        """Process real-time transcription updates ONLY for key commands."""
        if self.debug:
            logger.debug(f"Real-time update received: {text}")

        self._clean_recent_words()

        # Only process commands in real-time updates
        cleaned_for_commands = self.non_alpha_pattern.sub(" ", text).lower()
        words = cleaned_for_commands.split()
        i = 0
        while i < len(words):
            for cmd_len in range(min(4, len(words) - i), 0, -1):
                potential_cmd = " ".join(words[i : i + cmd_len])
                if potential_cmd in self.word_to_key_map and self._can_process_word(potential_cmd):
                    key_combo = self.word_to_key_map[potential_cmd]
                    self.key_press_queue.put(key_combo)
                    if self.on_command_processed:
                        self.on_command_processed(potential_cmd, key_combo)
                    i += cmd_len
                    break
            else:
                i += 1

    def _process_text(self, text: str) -> None:
        """Process the final transcription handling both commands and non-command text."""
        if self.debug:
            logger.debug(f"Final transcription received: {text}")

        # First clean punctuation, then split into words for command detection
        cleaned_for_commands = self.non_alpha_pattern.sub(" ", text).lower()
        words = [w.strip() for w in cleaned_for_commands.split() if w.strip()]

        # Check for commands in sequence of words
        i = 0
        processed_until = 0

        while i < len(words):
            for cmd_len in range(min(4, len(words) - i), 0, -1):
                potential_cmd = " ".join(words[i:i + cmd_len])
                if potential_cmd in self.word_to_key_map:
                    if self.debug:
                        logger.debug(f"Found command: {potential_cmd}")
                    
                    key_combo = self.word_to_key_map[potential_cmd]
                    self.key_press_queue.put(key_combo)
                    
                    if self.on_command_processed:
                        self.on_command_processed(potential_cmd, key_combo)
                    
                    processed_until = i + cmd_len
                    i += cmd_len
                    break
            else:
                i += 1

        # Process remaining text after commands (if any)
        if processed_until < len(words):
            remaining_text = text
            for word in words[:processed_until]:
                word_pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
                match = word_pattern.search(remaining_text)
                if match:
                    remaining_text = remaining_text[match.end():].lstrip()

            # Apply word transformations
            transformed_text = remaining_text
            for word, replacement in self.word_to_word_map.items():
                # Match the word with any surrounding punctuation
                pattern = re.compile(rf'\b{word}\b[^\s\w]*', re.IGNORECASE)
                transformed_text = pattern.sub(replacement, transformed_text)
                
            # Queue the transformed text for typing if it's not empty
            if transformed_text.strip():
                if self.debug:
                    logger.debug(f"Queueing transformed text: {transformed_text}")
                self.typing_queue.put(transformed_text.strip())

    def _voice_worker(self) -> None:
        """Background thread for voice recognition."""
        try:
            while True:
                self.recorder.text(on_transcription_finished=self._process_text)
        except KeyboardInterrupt:
            self.stop()

    def start(self) -> None:
        """Start the voice control system."""
        self.key_press_thread.start()
        self.typing_thread.start()

        logger.info("Voice control system started")
        logger.info("Available voice commands:")
        for word, key in self.word_to_key_map.items():
            logger.info(f"  Say '{word}' to press '{key}'")

        # Start voice recognition in a background thread
        self.voice_thread = threading.Thread(target=self._voice_worker, daemon=True)
        self.voice_thread.start()

    def stop(self) -> None:
        """Stop the voice control system and clean up resources."""
        if self.debug:
            logger.debug("Stopping voice control system...")

        if self.recorder:
            self.recorder.shutdown()

        # Signal workers to stop
        self.key_press_queue.put(None)
        self.typing_queue.put(None)

        # Wait for threads to finish
        if self.key_press_thread.is_alive():
            self.key_press_thread.join()
        if self.typing_thread.is_alive():
            self.typing_thread.join()
        if self.voice_thread and self.voice_thread.is_alive():
            self.voice_thread.join()

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


# Example usage:
if __name__ == "__main__":
    # Example command mapping
    WORD_TO_KEY_MAP = {
        "left": "left",
        "right": "right",
        "up": "up",
        "down": "down",
        "enter": "enter",
        "select all": ["ctrl", "a"],
        "copy": ["ctrl", "c"],
        "paste": ["ctrl", "v"],
        "delete": "delete",
    }
    WORD_TO_WORD_MAP = {"exit": "exit()", "print": "print()"}

    # Optional callback function
    def on_command(word, key):
        logger.info(f"Command processed: {word} -> {key}")

    # Create and start voice control
    vc = VoiceControl(
        word_to_key_map=WORD_TO_KEY_MAP, 
        word_to_word_map=WORD_TO_WORD_MAP, 
        debug=True, 
        on_command_processed=on_command
    )

    try:
        vc.start()
    except KeyboardInterrupt:
        vc.stop()
