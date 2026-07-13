# Brainfuck Small Language Model (SLM)

> *Technically* a language model. Please do not put this on your resume.

A character-level bigram/trigram language model written entirely in Brainfuck, with a Python interpreter, a fast C interpreter, and a unified Python training pipeline. Now with a **Nigerian Pidgin** edition.

## What Is This?

A tiny autoregressive language model that:
1. Takes a seed character (or pair) as input
2. Predicts the next character using hardcoded bigram/trigram transition probabilities
3. Feeds the prediction back as input and repeats
4. Outputs a generated sequence

## Files

| File | Description |
|------|-------------|
| `train.py` | Unified training pipeline — bigram & trigram modes |
| `interpreter.py` | Python Brainfuck interpreter with debug mode |
| `bf.c` | Fast C Brainfuck interpreter |
| `bfcc.py` | BF to C compiler + gcc runner |
| `corpus/english.txt` | English training corpus |
| `corpus/pidgin.txt` | Nigerian Pidgin training corpus (~50K chars) |
| `slm.bf` | Pre-trained English bigram model |
| `slm_pidgin.bf` | Pre-trained Pidgin trigram model |
| `slm_pidgin_v4.bf` | V4 trigram with bigger corpus and entropy |

## Install

### C interpreter (recommended)
```bash
gcc -O3 -o bf bf.c
```

### Python interpreter
No install needed — just run.

## Usage

### Run pre-trained models
```bash
echo -n "t" | ./bf slm.bf                    # English bigram
echo -n "de" | ./bf slm_pidgin_v4.bf          # Pidgin trigram
```

### Train new models
```bash
# English bigram
python3 train.py --corpus-file corpus/english.txt --mode bigram -o slm.bf

# Pidgin trigram
python3 train.py --corpus-file corpus/pidgin.txt --mode trigram -o slm_pidgin.bf

# Custom inline corpus
python3 train.py --corpus "hello world " --mode bigram -o custom.bf
```

### train.py options
```
--corpus-file FILE    Path to corpus text file
--corpus TEXT         Inline corpus string
--mode bigram|trigram Model mode (default: bigram)
-k, --candidates N    Max candidates per trigram context (default: 3)
-s, --steps N         Generation steps (default: 20 bigram, 60 trigram)
-o, --output FILE     Output .bf file
--verify              Print simulated outputs
```

### Debug mode
```bash
echo -n "t" | python3 interpreter.py slm.bf --debug
```

### BF to C compiler
```bash
python3 bfcc.py slm.bf                # compile + run
python3 bfcc.py slm.bf --c-only       # emit .c file
python3 bfcc.py slm.bf --compile-only  # compile to binary
```

## How It Works

### Bigram Model
Predicts the next character based solely on the current character:
- `t` → `h` (most common follower)
- `h` → `e`
- `e` → ` ` (space)

### Trigram Model
Uses 2-character context for better predictions:
- `('d','e')` → `y` ("de" → "dey")
- `(' ','a')` → `b` (" a" → "abeg")

### Temperature/Entropy
The trigram model stores top-K candidates per context and uses entropy-based selection to break deterministic loops, producing more natural-sounding output.

## Performance

| Model | Python | C | Speedup |
|-------|--------|---|---------|
| Bigram | ~0.5s | ~0.01s | ~50x |
| Trigram | ~30s+ | ~1-3s | ~30x |

## Fun Facts

- This is technically autoregressive generation — the same paradigm as GPT
- GPT has ~175 billion parameters. This has ~17 (bigram) or ~400 (trigram)
- The entire bigram model fits in 7 cells of a 60,000-cell tape
- Each trigram generation step runs millions of BF operations

## Disclaimer

This is *technically* a language model in the same way an abacus is *technically* a computer. It demonstrates that the core idea behind modern LLMs — predicting the next token — can be implemented in literally any Turing-complete system, including one with only 8 instructions.

Do not cite this in your ML paper. Do not put it on your resume. Do tell your friends about it at parties.
