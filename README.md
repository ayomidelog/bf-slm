# Brainfuck Small Language Model (SLM)

> *Technically* a language model. Please do not put this on your resume.

A character-level bigram/trigram language model written entirely in Brainfuck, with a Python interpreter, a fast C interpreter, and a Python "training pipeline" to generate it. Now with a **Nigerian Pidgin** edition.

## What Is This?

A tiny autoregressive language model that:
1. Takes a seed character (or pair) as input
2. Predicts the next character using hardcoded bigram/trigram transition probabilities
3. Feeds the prediction back as input and repeats
4. Outputs a generated sequence

## Files

| File | Description |
|------|-------------|
| `slm.bf` | Original bigram model trained on English (~7K instructions) |
| `slm_pidgin.bf` | Trigram model trained on Nigerian Pidgin (~277K instructions) |
| `slm_pidgin_v4.bf` | V4 trigram with bigger corpus and improved entropy (~370K instructions) |
| `interpreter.py` | Python Brainfuck interpreter with debug mode |
| `bf.c` | Fast C Brainfuck interpreter (~100x faster than Python) |
| `train.py` | Bigram training pipeline: corpus → bigram table → BF code |
| `train_slm_v4.py` | V4 trigram trainer with temperature/entropy |
| `bfcc.py` | BF to C compiler + gcc runner |
| `README.md` | You're reading it |

## Install

### Python interpreter
```bash
echo -n "t" | python3 interpreter.py slm.bf
```

### C interpreter (fast)
```bash
gcc -O3 -o bf bf.c
echo -n "t" | ./bf slm.bf
```

## Usage

### Run the English bigram model
```bash
echo -n "t" | ./bf slm.bf       # → "he the the the the t"
echo -n "c" | ./bf slm.bf       # → "athe the the the the"
```

### Run the Pidgin trigram model
```bash
echo -n "de" | ./bf slm_pidgin.bf    # → "y this ife i dey thell well..."
echo -n "we" | ./bf slm_pidgin.bf    # → "ll wan you dey to fol well..."
```

### Run the V4 Pidgin model
```bash
echo -n "de" | ./bf slm_pidgin_v4.bf
echo -n "ab" | ./bf slm_pidgin_v4.bf
```

### Retrain with a custom corpus
```bash
python3 train.py --corpus "your text here"
```

### Debug mode
```bash
echo -n "t" | python3 interpreter.py slm.bf --debug
```

## Performance

| Model | Python | C | Speedup |
|-------|--------|---|---------|
| Bigram (7K instructions) | 0.485s | 0.011s | 44x |
| Trigram Pidgin (277K instructions) | 30.0s | 0.3s | 100x |

## How It Works

### Bigram Model
Predicts the next character based solely on the current character. From the training corpus:
- `t` → `h` (most common follower)
- `h` → `e`
- `e` → ` ` (space)
- ` ` → `t`

### Trigram Model
Uses 2-character context (previous + current) for better predictions:
- `('w','e')` → `l` ("we" → "well")
- `('d','e')` → `y` ("de" → "dey")
- `(' ','d')` → `e` (" d" → "dey")

### Temperature/Entropy
V4 stores top-3 candidates per context and uses entropy-based selection to break deterministic loops.

### C Interpreter
- Comment-aware instruction loader (strips `;` lines)
- Optimized bracket map precomputation
- Direct `unsigned char` tape operations
- ~100x faster than the Python interpreter

## Fun Facts

- The bigram program has **7,915 instructions** encoding 17 transitions
- The trigram pidgin program has **277,904 instructions** encoding 385 transitions with temperature
- Each generation step in the trigram model runs millions of BF operations
- The entire bigram model fits in 7 cells of a 60,000-cell tape
- This is technically autoregressive generation, the same paradigm as GPT
- GPT has ~175 billion parameters. This has 17 (bigram) or 385 (trigram).

## Disclaimer

This is *technically* a language model in the same way an abacus is *technically* a computer. It demonstrates that the core idea behind modern LLMs — predicting the next token — can be implemented in literally any Turing-complete system, including one with only 8 instructions.

Do not cite this in your ML paper. Do not put it on your resume. Do tell your friends about it at parties.
