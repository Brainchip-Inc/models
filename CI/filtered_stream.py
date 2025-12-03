#!/usr/bin/env python3
"""Low-level stream filtering helpers.

Provides a context manager `redirect_and_filter_fds` that redirects the
process-level stdout/stderr file descriptors (fd 1/2) into pipes and
filters lines using regular expressions in real time. This captures C/C++
level output (TensorFlow/CUDA) that bypasses Python's `sys.stdout`.

Also includes convenience helpers to silence TensorFlow using env vars
and the Python logging API. Example usage is in the module docstring.
"""
import os
import sys
import threading
import re
from contextlib import contextmanager
import logging


def _reader_thread(pipe_fd, orig_fd, allow_pattern, deny_pattern, encoding='utf-8'):
    buf = ''
    try:
        while True:
            try:
                data = os.read(pipe_fd, 4096)
            except OSError as e:
                # If pipe_fd was closed from another thread (race or debugger),
                # exit gracefully.
                break
            if not data:
                break
            text = data.decode(encoding, errors='replace')
            buf += text
            lines = buf.splitlines(keepends=True)
            # If last line doesn't end with newline, keep it in buffer
            if lines and not lines[-1].endswith('\n'):
                buf = lines.pop()
            else:
                buf = ''
            for line in lines:
                l = line.strip()
                write_line = True
                # Allow-list wins
                if allow_pattern and allow_pattern.search(l):
                    write_line = True
                else:
                    if deny_pattern and deny_pattern.search(l):
                        write_line = False
                    else:
                        write_line = True

                if line == '\n' or write_line:
                    try:
                        os.write(orig_fd, line.encode(encoding, errors='replace'))
                    except OSError:
                        pass
    finally:
        # Flush any leftover partial line
        if buf:
            l = buf.strip()
            write_line = True
            if allow_pattern and allow_pattern.search(l):
                write_line = True
            else:
                if deny_pattern and deny_pattern.search(l):
                    write_line = False
                else:
                    write_line = True

            if write_line:
                try:
                    os.write(orig_fd, buf.encode(encoding, errors='replace'))
                except OSError:
                    pass
        # Ensure the pipe fd is closed by this thread to avoid fd leaks.
        try:
            os.close(pipe_fd)
        except Exception:
            pass


@contextmanager
def redirect_and_filter_fds(stdout_pattern=None, stderr_pattern=None):
    """Redirect process stdout/stderr (fd 1 and 2) into pipes and filter.

    - This intercepts output written at the OS/C level (C libraries, native
      extensions) which bypass Python's `sys.stdout`/`sys.stderr`.
    - Patterns may be provided as compiled `re.Pattern` or strings.
    - Filters are applied per-line; partial lines are buffered until newline.

    Example:
        from tools.stream_filter import redirect_and_filter_fds
        import re

        pattern = re.compile(r'(Checking model:|Loading model:)|^(?!.*(cuda|TensorFlow))', re.IGNORECASE)
        with redirect_and_filter_fds(stdout_pattern=pattern, stderr_pattern=pattern):
            import tensorflow as tf  # logs from C++ will be filtered

    Note: for best results set environment variables like `TF_CPP_MIN_LOG_LEVEL`
    before importing TensorFlow (see `set_tf_silent`).
    """
    orig_stdout_fd = sys.stdout.fileno()
    orig_stderr_fd = sys.stderr.fileno()

    # Save duplicates of the original fds so we can restore them later
    saved_stdout = os.dup(orig_stdout_fd)
    saved_stderr = os.dup(orig_stderr_fd)

    # Create pipes for capturing
    r_out, w_out = os.pipe()
    r_err, w_err = os.pipe()

    # Optionally detect other fds that point to the same underlying file as
    # stdout/stderr (e.g. duplicates) and redirect them to our pipe as well.
    replaced_fds = {}
    def _maybe_replace_matching_fds(orig_fd, write_end):
        try:
            orig_stat = os.fstat(orig_fd)
        except OSError:
            return

        # List candidate fds from /proc/self/fd (Linux-specific)
        try:
            for name in os.listdir('/proc/self/fd'):
                try:
                    fd = int(name)
                except Exception:
                    continue
                if fd in (orig_fd,):
                    continue
                # skip our own pipe ends and saved duplicates
                if fd in (r_out, w_out, r_err, w_err):
                    continue
                try:
                    stat = os.fstat(fd)
                except OSError:
                    continue
                # Match by device+inode (same underlying target)
                if stat.st_ino == orig_stat.st_ino and stat.st_dev == orig_stat.st_dev:
                    try:
                        # Save a duplicate so we can restore later
                        saved = os.dup(fd)
                        os.dup2(write_end, fd)
                        replaced_fds[fd] = saved
                    except OSError:
                        # ignore failures to duplicate/dup2
                        continue
        except Exception:
            # If /proc isn't available or listing fails, skip this step
            pass

    # Try to replace other fds that point to the same underlying target
    # before we overwrite stdout/stderr. This ensures writes to those fds
    # are also captured by our pipes.
    try:
        _maybe_replace_matching_fds(orig_stdout_fd, w_out)
    except Exception:
        pass
    try:
        _maybe_replace_matching_fds(orig_stderr_fd, w_err)
    except Exception:
        pass

    # Replace process stdout/stderr with pipe write ends
    os.dup2(w_out, orig_stdout_fd)
    os.dup2(w_err, orig_stderr_fd)

    # Close the duplicated write ends in this context (fds are now at 1/2)
    os.close(w_out)
    os.close(w_err)

    # Default pattern: keep most normal lines but exclude common noisy keywords
    # Default allow/deny patterns
    DEFAULT_ALLOW = re.compile(
        r'('
        r'Checking model:|'
        r'Loading model:|'
        r'✅.*needs \d+ Akida nodes'
        r')',
        re.IGNORECASE,
    )

    DEFAULT_DENY = re.compile(
        r'('
        r'cuda|'
        r'cufft|'
        r'cudnn|'
        r'cublas|'
        r'tensorflow|'
        r'xla|'
        r'absl::InitializeLog|'
        r'Unable to register|'
        r'WARNING:|'
        r'failed call to|'
        r'loop_optimizer\.cc|'
        r'computation_placer|'
        r'computation placer|'
        r'gpu_device|'
        r'Created device|'
        r'TF_FORCE_GPU_ALLOW_GROWTH|'
        r'Overriding orig_value'
        r')',
        re.IGNORECASE,
    )

    # If user provided a single pattern via stdout_pattern/ stderr_pattern,
    # treat it as a DENY override for convenience. Otherwise use defaults.
    def _ensure_patterns(user_pat):
        if user_pat is None:
            return (DEFAULT_ALLOW, DEFAULT_DENY)
        if isinstance(user_pat, str):
            return (DEFAULT_ALLOW, re.compile(user_pat, re.IGNORECASE))
        # if compiled pattern given, interpret as deny pattern
        return (DEFAULT_ALLOW, user_pat)

    out_allow, out_deny = _ensure_patterns(stdout_pattern)
    err_allow, err_deny = _ensure_patterns(stderr_pattern)

    # Start reader threads that forward allowed lines to the saved original fds
    t_out = threading.Thread(target=_reader_thread, 
        args=(r_out, saved_stdout, out_allow, out_deny), daemon=True)
    t_err = threading.Thread(target=_reader_thread, 
        args=(r_err, saved_stderr, err_allow, err_deny), daemon=True)
    t_out.start()
    t_err.start()

    try:
        yield
    finally:
        # Try flushing Python-level buffers first
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        # Restore original fds
        os.dup2(saved_stdout, orig_stdout_fd)
        os.dup2(saved_stderr, orig_stderr_fd)

        # Restore any other fds we replaced
        try:
            for fd, saved in list(replaced_fds.items()):
                try:
                    os.dup2(saved, fd)
                except OSError:
                    pass
                try:
                    os.close(saved)
                except OSError:
                    pass
        except Exception:
            pass

        # Close saved copies (these were duplicates of original fds)
        for fd in (saved_stdout, saved_stderr):
            try:
                os.close(fd)
            except OSError:
                pass

        # Wait a short time for threads to finish; threads will close their
        # read ends when they detect EOF or an OSError. Joining reduces races
        # with debuggers stepping through code.
        t_out.join(timeout=1.0)
        t_err.join(timeout=1.0)


def set_tf_silent(level: str = '2'):
    """Set TensorFlow/C++ logging environment and Python logging level.

    - `level` is the string value for `TF_CPP_MIN_LOG_LEVEL` ("0","1","2","3").
    - Call this BEFORE importing TensorFlow for earliest effect.
    """
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', level)
    try:
        import tensorflow as tf  # noqa: F401
        tf.get_logger().setLevel('ERROR')
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
    except Exception:
        # TensorFlow not available or import triggers logs; env var still helps
        pass
