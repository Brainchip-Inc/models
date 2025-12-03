#!/usr/bin/env python3
"""Simple post-capture filter for noisy commands.
Runs a subprocess, captures stdout+stderr, and prints allowed lines to the
console with a small configurable delay. This is useful in CI where
low-level fd redirection may not work.
Usage: 
    python CI/post_filter.py -- CI/check_model.py --models $FILES
The `--` separates options to this script from the command to run.
"""
import os
import sys
import time
import argparse
import subprocess
import re

def run_and_filter(cmd, delay=0.05):

    allow = re.compile(r'(Checking model:|Loading model:|✅.*needs \d+ Akida nodes)', re.IGNORECASE)
    deny = re.compile(
        r'(cuda|cufft|cudnn|cublas|tensorflow|xla|absl::InitializeLog|Unable to register|WARNING:|'
        r'failed call to|loop_optimizer\.cc|computation_placer|computation placer|gpu_device|Created device|'
        r'TF_FORCE_GPU_ALLOW_GROWTH|Overriding orig_value)'
        , re.IGNORECASE)        
    
    # If the command is a Python script path and not executable, prefix
    # with the current Python interpreter to avoid PermissionError on CI.
    if isinstance(cmd, (list, tuple)) and cmd:
        first = cmd[0]
        if isinstance(first, str) and first.endswith('.py') and not os.access(first, os.X_OK):
            cmd = [sys.executable] + list(cmd)
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    try:
        # Read lines as they arrive and print with a small delay
        for raw in iter(proc.stdout.readline, ''):
            if raw is None:
                break
            line = raw.rstrip('\n')
            keep = True
            if allow.search(line):
                keep = True
            else:
                if deny.search(line):
                    keep = False
                else:
                    keep = True

            # Small delay before printing
            if delay and delay > 0:
                time.sleep(delay)

            if keep:
                print(line, flush=True)

        proc.wait()
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass

def main(argv=None):
    p = argparse.ArgumentParser(prog='post_filter', description='Run command and filter its output with a small delay')
    p.add_argument('--delay', type=float, default=0.05, help='Delay in seconds before printing each line')
    p.add_argument('--deny', type=str, default=None, help='Optional deny regex (overrides default)')
    p.add_argument('--allow', type=str, default=None, help='Optional allow regex (overrides default)')
    p.add_argument('cmd', nargs=argparse.REMAINDER, help='Command to run (use -- before the command)')
    args = p.parse_args(argv)

    if not args.cmd:
        print('No command provided.' 
              'Use -- to separate command, e.g.: python CI/post_filter.py -- CI/check_model.py --models $FILES')
        return 2

    # Remove leading '--' if present
    cmd = args.cmd
    if cmd and cmd[0] == '--':
        cmd = cmd[1:]

    run_and_filter(cmd)

if __name__ == '__main__':
    sys.exit(main())
