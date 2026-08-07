# Acknowledgements

This benchmark stands on other people's open-source work. Thank you.

## Audio8 TTS

The synthetic speech in `audio_synthetic/` was generated with
[**Audio8-TTS-Preview-0.6b**](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b)
([code](https://github.com/Audio8-AI/Audio8_TTS), Apache License 2.0 for both
code and weights).

A sincere thank-you to the Audio8 team for releasing it openly. Finding a good
open-source German TTS is genuinely hard — most options are either
English-first, with German as an afterthought that mangles compound words and
umlauts, or they are not open at all. Audio8 was the best German-capable
open-source model found while building this benchmark, and it is the reason the
synthetic audio channel exists without depending on a commercial API.

## Whisper

Every WAV in this dataset is verified by ASR round-trip, and spoken answers are
transcribed for scoring, using OpenAI's Whisper large-v3 in Apple's MLX port:
[`mlx-community/whisper-large-v3-mlx`](https://huggingface.co/mlx-community/whisper-large-v3-mlx)
(MIT), run via [`mlx-whisper`](https://github.com/ml-explore/mlx-examples)
(MIT, © 2023 Apple Inc.).

## Product names

The product names this dataset is built on are real dm-drogerie markt products,
taken from the publicly accessible product search on dm.de.

Everything else — suppliers, warehouses, regions, buyers, teams, purchase
prices, calorie values, stock levels, and every gold answer — is invented and
generated deterministically from a seed, and corresponds to no real company,
location, person, or commercial figure.

## Human speakers

The recordings in `audio_real/` were read by two German speakers who gave their
time to read 50 questions each, verbatim, in one sitting. Thank you.
