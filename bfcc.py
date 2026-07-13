#!/usr/bin/env python3
"""
bfcc.py — Brainfuck to C compiler + gcc runner.
"""

import argparse
import os
import subprocess
import sys


def load_program(path: str) -> str:
    """Load a .bf file, stripping comments and non-instruction characters."""
    valid = set("><+-.,[]")
    with open(path, "r") as f:
        lines = f.readlines()
    code_lines = [l for l in lines if not l.lstrip().startswith(";")]
    return "".join(c for c in "".join(code_lines) if c in valid)


def compile_to_c(program: str, tape_size: int = 30000) -> str:
    """Compile BF to C code, with merge optimization for runs."""
    body_parts = []
    i = 0
    while i < len(program):
        c = program[i]
        if c in "+-><":
            count = 0
            while i < len(program) and program[i] == c:
                count += 1
                i += 1
            if c == "+":
                body_parts.append(f"tape[p]+={count};" if count > 1 else "tape[p]++;")
            elif c == "-":
                body_parts.append(f"tape[p]-={count};" if count > 1 else "tape[p]--;")
            elif c == ">":
                body_parts.append(f"p+={count};" if count > 1 else "p++;")
            elif c == "<":
                body_parts.append(f"p-={count};" if count > 1 else "p--;")
        elif c == "[":
            body_parts.append("while(tape[p]){")
            i += 1
        elif c == "]":
            body_parts.append("}")
            i += 1
        elif c == ".":
            body_parts.append("putchar(tape[p]);")
            i += 1
        elif c == ",":
            body_parts.append("{int b=getchar();tape[p]=(b==EOF)?0:b;}")
            i += 1
        else:
            i += 1

    body = "\n".join(body_parts)

    return f"""#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {{
    unsigned char tape[{tape_size}];
    int p = 0;
    memset(tape, 0, {tape_size});

    {body}

    return 0;
}}
"""


def main():
    parser = argparse.ArgumentParser(description="BF to C compiler + runner")
    parser.add_argument("program", help="Path to .bf file")
    parser.add_argument("-o", "--output", help="Output binary name")
    parser.add_argument("--compile-only", action="store_true", help="Just compile, don't run")
    parser.add_argument("--c-only", action="store_true", help="Just emit .c file")
    parser.add_argument("-t", "--tape-size", type=int, default=30000)
    args = parser.parse_args()

    bf_code = load_program(args.program)
    if not bf_code:
        print("Error: empty program", file=sys.stderr)
        sys.exit(1)

    c_code = compile_to_c(bf_code, args.tape_size)

    base = os.path.splitext(os.path.basename(args.program))[0]
    c_path = f"{base}.c"
    with open(c_path, "w") as f:
        f.write(c_code)

    print(f"Compiled {len(bf_code)} BF instructions -> {c_path}")

    if args.c_only:
        return

    binary = args.output or base
    gcc_cmd = ["gcc", "-O2", "-o", binary, c_path]
    print(f"{' '.join(gcc_cmd)}")
    result = subprocess.run(gcc_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"gcc failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    print(f"Binary ready: ./{binary}")

    if args.compile_only:
        return

    print(f"Running {binary}...\n")
    subprocess.run([f"./{binary}"])


if __name__ == "__main__":
    main()
