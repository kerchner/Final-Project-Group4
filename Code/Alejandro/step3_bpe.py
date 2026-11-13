"""Step 3 - Train our own BPE tokenizers (vocabulary sizes 512 and 4,096) on the training sequences.

Run:  python step3_bpe.py
Saves models/bpe512.json and models/bpe4096.json (used by step 4).
"""
from tokenizers import Tokenizer, models, trainers

from common import MODELS_DIR, load_data

train, dev, test = load_data()

for vocab_size in [512, 4096]:
    # No pre-tokenizer: DNA has no spaces, so BPE learns chunks directly from the 300-letter strings
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]"],   # ids 0 and 1
        initial_alphabet=list("ACGT"),
        show_progress=False,
    )
    tokenizer.train_from_iterator(train.sequence, trainer)   # train split only, no peeking at dev/test
    path = MODELS_DIR / f"bpe{vocab_size}.json"
    tokenizer.save(str(path))

    # Quick look at the result
    lengths = [len(tokenizer.encode(s).ids) for s in dev.sequence]
    example = tokenizer.encode(dev.sequence[0][:60]).tokens
    print(f"\nBPE {vocab_size}: saved to {path}")
    print(f"  average tokens per 300-letter sequence: {sum(lengths) / len(lengths):.1f}")
    print(f"  first 60 letters of a dev sequence -> {example}")
