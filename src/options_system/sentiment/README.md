# sentiment/

**Scores news/text into a numeric sentiment feature for the model.** Baseline
is **FinBERT** (`transformers` + `torch`) running on the local GPU, classifying
finance text (headlines, filings, macro releases) into positive/negative/
neutral with a confidence score. There is an optional upgrade path to a local
~8B LLM (Qwen3-8B / Llama-3.1-8B) served via **Ollama** if a richer signal is
warranted. Crucially, sentiment scoring is part of the **offline / feature**
side and the inference is cheap and local — **no paid LLM API and no LLM in
the live trading decision loop**. Output is a plain timestamped sentiment value
the `features` module can join to price data point-in-time.
