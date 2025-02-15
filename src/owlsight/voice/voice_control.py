from RealtimeSTT import AudioToTextRecorder
import pyautogui
import re
import threading
import queue
import time
from typing import Dict, Optional, Callable
from owlsight.utils.logger import logger


class VoiceControl:
    """
    A class to handle voice-controlled keyboard input with real-time speech recognition.
    Supports command words that trigger key presses and types other spoken words.
    """

    def __init__(
        self,
        word_to_key_map: Dict[str, str],
        word_cooldown: float = 0.8,
        debug: bool = False,
        language: str = "en",
        model: str = "small.en",
        key_press_interval: float = 0.05,
        typing_interval: float = 0.03,
        on_command_processed: Optional[Callable[[str, str], None]] = None,
    ):
        """
        Initialize the VoiceControl instance.

        Parameters:
        ----------
            word_to_key_map: Dictionary mapping command words to keyboard keys
            word_cooldown: Cooldown period (in seconds) between same command recognition
            debug: Enable debug printing
            language: Language for speech recognition
            model: Model to use for speech recognition
            key_press_interval: Interval between key presses
            typing_interval: Interval between typing characters
            on_command_processed: Optional callback when a command is processed
        """
        self.word_to_key_map = word_to_key_map
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
        self.words_to_filter = set(word_to_key_map.keys())

        # Compile regex patterns
        self.non_alpha_pattern = re.compile(r"[^a-zA-Z\s]")
        self.filter_pattern = re.compile(r"\b(" + "|".join(map(re.escape, self.words_to_filter)) + r")\b")

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
            key = self.key_press_queue.get()
            if key is None:
                break
            pyautogui.press(key, interval=self.key_press_interval)
            if self.debug:
                logger.debug(f"Pressed key: {key}")
            self.key_press_queue.task_done()

    def _typing_worker(self) -> None:
        """Worker thread for processing text typing."""
        if self.debug:
            logger.debug("Typing worker started")
        while True:
            text = self.typing_queue.get()
            if text is None:
                break
            for word in text.split():
                if word.lower() not in self.word_to_key_map:
                    pyautogui.write(word + " ", interval=self.typing_interval)
                    if self.debug:
                        logger.debug(f"Typed word: {word}")
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
        """Process real-time transcription updates and trigger key presses."""
        if self.debug:
            logger.debug(f"trigger_keys received: {text}")

        self._clean_recent_words()

        for word in text.split():
            cleaned_word = self.non_alpha_pattern.sub("", word).lower()
            if self.debug:
                logger.debug(f"Processed word: {cleaned_word}")

            if cleaned_word in self.word_to_key_map and self._can_process_word(cleaned_word):
                key = self.word_to_key_map[cleaned_word]
                self.key_press_queue.put(key)
                if self.on_command_processed:
                    self.on_command_processed(cleaned_word, key)
                if self.debug:
                    logger.debug(f"Queued key: {key} for word: {cleaned_word}")

    def _process_text(self, text: str) -> None:
        """Process the final transcription for typing non-command words."""
        # First clean any non-alpha characters and convert to lowercase for matching
        cleaned_text = self.non_alpha_pattern.sub(" ", text).lower()
        
        # Split into words and filter out command words
        words = []
        for word in cleaned_text.split():
            if word not in self.word_to_key_map:
                # Find the original word with its punctuation from the input text
                # by matching position
                start_idx = text.lower().find(word)
                if start_idx != -1:
                    # Get the original word with its case and punctuation
                    end_idx = start_idx + len(word)
                    original_word = text[start_idx:end_idx]
                    words.append(original_word)

        # Join filtered words back together
        final_text = " ".join(words)
        
        if self.debug:
            logger.debug(f"Final transcription after filtering: {final_text}")
        
        if final_text.strip():  # Only queue if there's actual text to type
            self.typing_queue.put(final_text)

    def _voice_worker(self) -> None:
        """Background thread for voice recognition."""
        try:
            while True:
                self.recorder.text(self._process_text)
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
    WORD_TO_KEY_MAP = {"boom": "left", "bang": "right", "ding": "up", "dong": "down", "sick": "enter"}

    # Optional callback function
    def on_command(word, key):
        logger.info(f"Command processed: {word} -> {key}")

    # Create and start voice control
    vc = VoiceControl(word_to_key_map=WORD_TO_KEY_MAP, debug=True, on_command_processed=on_command)

    try:
        vc.start()
    except KeyboardInterrupt:
        vc.stop()
