#!/usr/bin/env python3
import sys
import re
from contextlib import contextmanager

class FilteredStream:
    """A custom stream filter that intercepts and filters output based on specific patterns.
    This class is designed to filter out unwanted log messages while allowing relevant 
    ones to pass through."""
    def __init__(self, original):
        """Initialize the FilteredStream with the original stream (stdout or stderr).
        Args:
            original: The original stream to wrap (e.g., sys.stdout or sys.stderr).
        """
        self.original = original
        # Compile a regex pattern to match relevant messages or exclude unwanted ones.
        self.pattern = re.compile(
            r'(Checking model:|Loading model:|✅.*needs \d+ Akida nodes)|'  # Allow specific relevant messages.
            r'^(?!.*(cuda|Cuda|CUDA|TensorFlow|tensorflow|XLA|xla|Unable to register|'  # Exclude unwanted keywords.
            r'WARNING:|failed call to|loop_optimizer\.cc|computation placer))' # Exclude unwanted keywords. (continued)
        )

    def write(self, text):
        """Write method to filter and forward text to the original stream if it matches the pattern.
        Args:
            text: The text to be written to the stream.
        Returns:
            int: The number of characters written.
        """
        if text and text.strip() and self.pattern.search(text.strip()):
            # Write the text to the original stream if it matches the pattern.
            self.original.write(text)
            self.original.flush()
        elif text == '\n':
            # Always write newline characters to maintain formatting.
            self.original.write(text)
            self.original.flush()
        return len(text)

    def flush(self):
        """Flush the original stream to ensure all data is written."""
        self.original.flush()

    def close(self):
        """Placeholder for closing the stream. No operation is performed here."""
        pass

    def isatty(self):
        """Check if the original stream is a TTY (interactive terminal).
        Returns:
            bool: True if the original stream is a TTY, False otherwise.
        """
        return self.original.isatty()

@contextmanager
def filtered_output():
    """Context manager redirects and filters the I/O"""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    sys.stdout = FilteredStream(original_stdout)
    sys.stderr = FilteredStream(original_stderr)
    
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
