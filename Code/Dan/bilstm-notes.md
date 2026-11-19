The frozen DNABERT-2 baseline finished with dev MCC 0.802 and test MCC 0.783, and the earlier 0.10 was a small-subset artifact. The BiLSTM runs are still training (epoch 3, dev MCC 0.829), and I'll keep checking on them. · summary

I've implemented the BiLSTM and the frozen DNABERT-2 baseline, and the frozen DNABERT-2 run has finished. The BiLSTM runs for all four tokenizers are still training in the background.

Frozen DNABERT-2 (finished)
- Method: dnabert2_frozen_train.py embeds every sequence once with the frozen model. It then fits a logistic regression on the 768-number vectors, and the embeddings are cached in results/.
- Dev selection: Pooling (cls or mean) and C were chosen on the dev split only. Mean pooling with C=0.001 won, at dev MCC 0.802.
- Test result: MCC 0.783, accuracy 0.892, F1 0.893, ROC-AUC 0.951. The test set was scored once.
- Loading fixes: DNABERT-2's published loading code doesn't work with transformers 5 on a Mac, so load_pretrained_dnabert2 works around it:
  - It skips the hard triton requirement.
  - It sets the missing pad_token_id.
  - It builds the model on CPU and loads the weights itself, because the meta-device loading broke the model's attention-bias buffer.
- App wrapper: DNABERT2FrozenModel is filled in for later use in the app.

BiLSTM (still training)
- Tokenizers: bilstm_model.py and bilstm_train.py support four tokenizers: single letters, overlapping 6-mers, BPE-512 and BPE-4096. The BPE tokenizers are trained on the training split only and stored inside each checkpoint.
- Network: The network ignores padding and summarizes the sequence from the final hidden states plus a max-pool over all positions.
- Training: It trains with early stopping on dev loss. It saves each run's dev metrics to results/bilstm_summary.csv, with checkpoints as results/bilstm_<tokenizer>.pt.
- Test scoring: --test loads a saved model and scores the test set without training, so the test set is touched once per final model.
- Progress: The char run is at epoch 3 with dev MCC 0.829, at roughly 6 minutes per epoch. The other three tokenizers run after it, so expect a couple more hours in total.
- Defaults: I used 1 layer, 64-dimensional embeddings, hidden size 128, dropout 0.3 and batch 128, instead of the stub's 2 layers, 128 and 256. Two layers were several times slower on this machine. The --layers, --emb and --hidden flags change this.
- App wrapper: BiLSTMModel can already load a checkpoint and predict, for when we wire it into the app.

Things to know
- Environment: I added pytorch, tokenizers, transformers and einops to pixi.toml.
- Overlap with Alejandro: Alejandro's step3_bpe.py, step4_bilstm.py and step5_dnabert_frozen.py already cover this ground, so our numbers will be a second run of the same experiments.
- Smoke test: I ran the BiLSTM script as a 2-epoch smoke test on all four tokenizers before launching the full runs.
- Untested: I haven't run --test for the BiLSTMs yet.
- Next step: When the BiLSTM runs finish, I'll report dev MCC per tokenizer. You can then choose which models get scored on the test set.
