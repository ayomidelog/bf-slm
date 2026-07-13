# Roadmap

## ✅ Done
- [x] Bigram character-level model (English)
- [x] Trigram model with 2-char context (Pidgin)
- [x] Temperature/entropy-based candidate selection
- [x] 50K char Nigerian Pidgin corpus
- [x] Python interpreter with debug mode
- [x] Fast C interpreter (~100x faster)
- [x] BF-to-C compiler (`bfcc.py`)
- [x] Unified training pipeline (`train.py`)
- [x] Corpus extraction to separate files

## 🔜 Next
- [ ] **Word-level model** — predict next WORD instead of next character. Small vocabulary (~500 words), Markov chain, much better output coherence.
- [ ] **N-gram model support** — configurable n-gram (bigram, trigram, quadgram) in training pipeline
- [ ] **Interactive REPL** — `./bf-repl slm_pidgin.bf` for live generation with seed input
- [ ] **Web demo** — Flask/Bun server + frontend, seed picker, real-time generation, tunneled with cftunnel
- [ ] **Larger corpora** — scrape real pidgin text from social media, Nollywood scripts, news

## 💡 Ideas
- [ ] **Beam search** — generate multiple candidate sequences, pick the best
- [ ] **Perplexity scoring** — measure how "good" the generated text is
- [ ] **Cross-domain training** — train on code, poetry, Shakespeare, medical text
- [ ] **BF bytecode compiler** — ahead-of-time compile .bf to native x86
- [ ] **WASM target** — compile BF interpreter to WebAssembly for browser play
- [ ] **Benchmark suite** — compare bigram vs trigram vs quadgram vs word-level across domains
- [ ] **Community corpus contributions** — let people add their own training data via PR
