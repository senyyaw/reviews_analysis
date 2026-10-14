"""Text preprocessing: one shared normalisation step, then two tracks.

Track A (model input) is cleaned minimally: case, punctuation, emoji, numbers and negations are
kept, because transformers use them. Track B (lexical, for keywords) is lowercased, lemmatised and
stopword-filtered with negations kept, and runs on English text, translated where needed.
See the README, Processing.
"""
