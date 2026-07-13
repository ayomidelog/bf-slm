#!/usr/bin/env python3
"""
Axon's Brainfuck Interpreter
=============================
Standard BF interpreter with debug mode.
Usage:
    python3 interpreter.py slm.bf
    python3 interpreter.py slm.bf --debug
    python3 interpreter.py slm.bf --tape-size 60000
"""

import argparse
import sys


def load_program(path: str) -> str:
    """Load a .bf file, stripping comments and non-instruction characters."""
    valid = set("><+-.,[]")
    with open(path, "r") as f:
        lines = f.readlines()
    # Strip comment lines (lines starting with ;)
    code_lines = [l for l in lines if not l.lstrip().startswith(";")]
    return "".join(c for c in "".join(code_lines) if c in valid)


def run(program: str, tape_size: int = 30000, debug: bool = False):
    """Execute a brainfuck program."""
    tape = [0] * tape_size
    ptr = 0
    ip = 0
    output_buf = []

    # Precompute bracket matching
    bracket_map = {}
    stack = []
    for i, c in enumerate(program):
        if c == "[":
            stack.append(i)
        elif c == "]":
            if not stack:
                raise RuntimeError(f"Unmatched ']' at instruction {i}")
            j = stack.pop()
            bracket_map[j] = i
            bracket_map[i] = j
    if stack:
        raise RuntimeError(f"Unmatched '[' at instruction {stack[-1]}")

    step = 0
    while ip < len(program):
        cmd = program[ip]

        if debug and step % 500 == 0:
            tape_window = tape[max(0, ptr - 3):ptr + 4]
            print(
                f"  step={step:>8}  ptr={ptr:>4}  ip={ip:>6}  "
                f"cmd='{cmd}'  tape[{max(0,ptr-3)}..{ptr+3}]={tape_window}",
                file=sys.stderr,
            )

        if cmd == ">":
            ptr += 1
            if ptr >= tape_size:
                raise RuntimeError(f"Pointer overflow at step {step}")
        elif cmd == "<":
            ptr -= 1
            if ptr < 0:
                raise RuntimeError(f"Pointer underflow at step {step}")
        elif cmd == "+":
            tape[ptr] = (tape[ptr] + 1) % 256
        elif cmd == "-":
            tape[ptr] = (tape[ptr] - 1) % 256
        elif cmd == ".":
            ch = chr(tape[ptr])
            output_buf.append(ch)
            # Flush every 64 chars so output appears progressively
            if len(output_buf) >= 64:
                sys.stdout.write("".join(output_buf))
                sys.stdout.flush()
                output_buf.clear()
        elif cmd == ",":
            # Read one byte from stdin
            b = sys.stdin.buffer.read(1)
            tape[ptr] = b[0] if b else 0
        elif cmd == "[":
            if tape[ptr] == 0:
                ip = bracket_map[ip]
        elif cmd == "]":
            if tape[ptr] != 0:
                ip = bracket_map[ip]

        ip += 1
        step += 1

    # Flush remaining output
    if output_buf:
        sys.stdout.write("".join(output_buf))
        sys.stdout.flush()

    if debug:
        print(f"\n  [debug] Done. Total steps: {step}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Brainfuck interpreter")
    parser.add_argument("program", help="Path to .bf file")
    parser.add_argument(
        "--tape-size", type=int, default=30000, help="Tape size (default: 30000)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug trace on stderr"
    )
    args = parser.parse_args()

    program = load_program(args.program)

    if not program:
        print("Warning: program is empty (no BF instructions found)", file=sys.stderr)
        return

    if args.debug:
        print(f"[debug] Loaded {len(program)} instructions from {args.program}", file=sys.stderr)

    try:
        run(program, tape_size=args.tape_size, debug=args.debug)
    except RuntimeError as e:
        print(f"\nRuntime error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[interrupted]", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
