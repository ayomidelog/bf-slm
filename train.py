#!/usr/bin/env python3
"""
train.py — The "Training Pipeline" for a Brainfuck Small Language Model
=========================================================================

Takes a text corpus, builds a character-level bigram transition table,
and emits a Brainfuck program that acts as an autoregressive language model.

Usage:
    python3 train.py                          # use default corpus
    python3 train.py --corpus "some text"     # custom corpus
    python3 train.py -o my_model.bf           # custom output path

The generated BF program:
  1. Reads a seed character from stdin
  2. Looks up the most-likely-next-character via conditional comparisons
  3. Outputs the predicted character
  4. Feeds it back as the new "current" character
  5. Repeats for 20 generation steps
"""

import argparse
import sys
from collections import Counter


# ─── The tiny training corpus ───────────────────────────────────────
DEFAULT_CORPUS = (
    "the cat sat on the mat and the dog ran to the park "
    "the cat liked the dog and the dog liked the cat "
    "she sat on the mat and he ran to the park "
    "the cat and the dog sat on the mat "
)


def build_bigram_table(corpus: str) -> dict[str, str]:
    """
    Build a bigram transition table from the corpus.
    For each character, find the most frequent character that follows it.
    Returns {current_char: most_likely_next_char}.
    """
    # Count all bigram pairs
    bigram_counts: dict[str, Counter] = {}
    for i in range(len(corpus) - 1):
        current = corpus[i]
        nxt = corpus[i + 1]
        if current not in bigram_counts:
            bigram_counts[current] = Counter()
        bigram_counts[current][nxt] += 1

    # For each character, pick the most common follower
    table = {}
    for char, followers in bigram_counts.items():
        most_common_char, count = followers.most_common(1)[0]
        table[char] = most_common_char

    return table


def compute_generation(table: dict[str, str], seed: str, steps: int = 20) -> str:
    """
    Simulate the model: starting from seed, follow bigram transitions.
    Returns the generated string (for verification).
    """
    result = [seed]
    current = seed
    for _ in range(steps):
        nxt = table.get(current, " ")
        result.append(nxt)
        current = nxt
    return "".join(result)


# ─── Brainfuck code generation ──────────────────────────────────────
#
# Tape layout:
#   Cell 0 : iteration counter (set to 20)
#   Cell 1 : current character (seed, then updated each step)
#   Cell 2 : comparison copy of current char
#   Cell 3 : comparison target value (V)
#   Cell 4 : match flag (1 = match found)
#   Cell 5 : predicted next character (written on match)
#   Cell 6-7 : temp for copying cell 1 → cell 2
#
# Comparison block for (V → P):
#   1. Copy cell 1 to cell 2 (preserving cell 1)
#   2. Set cell 3 = V (ASCII of vocab char)
#   3. [- >- <] : simultaneous decrement (comparison)
#   4. If both cell 2 and cell 3 are 0 → match!
#   5. Flag logic: cell 4 starts at 1, cleared if any mismatch
#   6. On match: cell 5 = P (predicted next char)
#   7. Restore pointer to cell 0
# ────────────────────────────────────────────────────────────────────


def bf_set_cell(value: int) -> str:
    """Generate BF code to set the current cell to a specific value.
    Uses a compact encoding: groups of 10 plus-signs."""
    if value == 0:
        return "[-]"
    parts = []
    while value >= 10:
        parts.append("++++++++++")
        value -= 10
    if value > 0:
        parts.append("+" * value)
    return "".join(parts)


def generate_comparison_block(vocab_char: str, predicted_char: str) -> str:
    """
    Generate a BF comparison block for one vocab character.
    
    Starting pointer position: cell 0
    Ending pointer position: cell 0
    
    If current_char == vocab_char, sets cell 5 = predicted_char.
    """
    V = ord(vocab_char)
    P = ord(predicted_char)
    lines = []

    # Step 1: Move to cell 1 and copy to cell 2 (using cell 6 as temp)
    # Copy cell 1 → cell 2, preserving cell 1
    lines.append(">")
    # [- >+ >>>>+ <<<<<] : dec cell1, inc cell2, inc cell6, back to cell1
    lines.append("[- >+ >>>>+ <<<<<]")
    # >>>>>[-<<<<<+>>>>>]<<<<< : move cell6 back to cell1
    lines.append(">>>>>[-<<<<<+>>>>>]<<<<<")
    # Now at cell 1. Cell 1=C, cell 2=C, cell 6=0.

    # Step 2: Set cell 3 = V
    lines.append(f">>{bf_set_cell(V)}<<")
    # At cell 1.

    # Step 3: Compare cell 2 with cell 3
    lines.append(">[- >- <]")
    # > to cell 2, [- >- <] decrement both. At cell 2.
    # After: cell2 = max(0,C-V), cell3 = max(0,V-C)

    # Step 4: Set flag at cell 4 = 1
    lines.append(">>[-]+<<")
    # >> to cell 4, set to 1. << to cell 2.

    # Step 5: Check cell 2 (if > 0 → C > V → not a match → clear flag)
    lines.append("[- >>[-]<< [-]]")
    # At cell 2: if nonzero, >> to 4 clear flag, << to 2 clear cell 2.

    # Step 6: Check cell 3 (if > 0 → C < V → not a match → clear flag)
    lines.append(">[- >[-]< [-]]")
    # > to cell 3: if nonzero, > to 4 clear flag, < to 3 clear cell 3.

    # Step 7: Check flag (cell 4) → if match, set cell 5 = P
    lines.append(">[- >[-]" + bf_set_cell(P) + "<]")
    # > to cell 4: if flag=1, clear flag, > to 5 set to P, < to 4.

    # Step 8: Back to cell 0
    lines.append("<<<<")
    # At cell 0.

    return "\n".join(lines)


def generate_program(table: dict[str, str]) -> str:
    """
    Generate the complete Brainfuck program.
    """
    # Sort vocab chars for deterministic output
    vocab = sorted(table.keys())

    # Filter to printable chars only (skip control chars)
    vocab = [c for c in vocab if 32 <= ord(c) <= 126]

    lines = []
    lines.append("; Brainfuck Small Language Model (SLM)")
    lines.append("; Character level bigram predictor")
    lines.append("; Generated by train.py")
    lines.append(";")
    lines.append("; Tape layout:")
    lines.append(";   cell 0 = iteration counter (20)")
    lines.append(";   cell 1 = current character")
    lines.append(";   cell 2 = comparison copy")
    lines.append(";   cell 3 = comparison target")
    lines.append(";   cell 4 = match flag")
    lines.append(";   cell 5 = prediction result")
    lines.append(";   cell 6 = copy temp")
    lines.append(";")
    lines.append(f"; Vocab: {' '.join(repr(c) for c in vocab)}")
    lines.append(";")

    # ── Init phase ──
    lines.append("; === INIT ===")
    # Set cell 0 = 20 (iteration counter)
    lines.append(f"; Set counter = 20")
    lines.append(bf_set_cell(20))
    # Move to cell 1 and read seed
    lines.append("> ,")
    lines.append("<")
    # At cell 0 after reading.

    # ── Main generation loop ──
    lines.append("")
    lines.append("; === GENERATION LOOP (20 iterations) ===")
    lines.append("[")
    lines.append("  -")
    lines.append("  ; Comparison chain")

    # Generate comparison blocks for each vocab char
    for v in vocab:
        p = table[v]
        lines.append(f"  ; '{v}' (ord={ord(v)}) → '{p}' (ord={ord(p)})")
        lines.append(generate_comparison_block(v, p))
        lines.append("")

    lines.append("  ; End of comparison chain")
    lines.append("")

    # After all comparisons: cell 5 has the prediction (or 0 if no match)
    # Move prediction to cell 1 and output
    lines.append("  ; Clear cell 1 and move prediction from cell 5 to cell 1")
    lines.append("  >[-] >>>>[-<<<<+>>>>]")
    # At cell 0, > to 1 clear, >>>> to 5, move to 1. At cell 5.

    lines.append("  ; Output prediction (cell 1)")
    lines.append("  <<<<.")
    # At cell 1, output. At cell 1.

    lines.append("  ; Back to cell 0 for loop check")
    lines.append("  <")
    # At cell 0.

    lines.append("]")
    lines.append("")
    lines.append("; === END ===")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Train a Brainfuck SLM from a text corpus"
    )
    parser.add_argument(
        "--corpus", type=str, default=DEFAULT_CORPUS, help="Training text corpus"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="slm.bf", help="Output BF file (default: slm.bf)"
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="Generation steps (default: 20)"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Print verification of the bigram model"
    )
    args = parser.parse_args()

    corpus = args.corpus

    # Build bigram table
    table = build_bigram_table(corpus)

    print(f"📊 Training on {len(corpus)} characters...", file=sys.stderr)
    print(f"📊 Vocabulary: {len(table)} unique chars", file=sys.stderr)
    print(f"📊 Bigram transitions:", file=sys.stderr)
    for char in sorted(table.keys()):
        if 32 <= ord(char) <= 126:
            print(
                f"     '{char}' → '{table[char]}' (ord {ord(char)} → {ord(table[char])})",
                file=sys.stderr,
            )

    # Verify with a few seeds
    if args.verify or True:
        print("", file=sys.stderr)
        for seed in ["t", "c", "d", "s", "p"]:
            gen = compute_generation(table, seed, args.steps)
            print(f"  seed='{seed}' → \"{gen}\"", file=sys.stderr)
        print("", file=sys.stderr)

    # Generate BF program
    program = generate_program(table)

    # Write to file
    with open(args.output, "w") as f:
        f.write(program)

    # Stats
    import re
    instruction_count = len(re.findall(r"[><+\-.,\[\]]", program))
    print(
        f"✅ Generated {args.output} ({instruction_count} BF instructions, {len(table)} vocab chars)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
