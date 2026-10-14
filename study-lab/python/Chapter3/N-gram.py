# e.g
# Dy has the wisdom
# Dy has the foresight
# total tokens: 8

import collections

corpus = "Dy has the wisdom Dy has the foresight"
tokens = corpus.split()
total_tokens = len(tokens)

unigrams_counts = collections.Counter(tokens)
bigrams = list(zip(tokens, tokens[1:]))
bigrams_counts = collections.Counter(bigrams)
trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
trigrams_counts = collections.Counter(trigrams)

# --- first step: calculate P(Dy) ---
count_dy = unigrams_counts["Dy"]
p_dy = count_dy / total_tokens
print(f"first step: P(Dy) = {count_dy} / {total_tokens} = {p_dy:.3f}")

# --- second step: calculate P(has|Dy) ---
count_dy_has = bigrams_counts[("Dy", "has")]
p_has_given_dy = count_dy_has / unigrams_counts["Dy"]
print(
    f"second step: P(has|Dy) = {count_dy_has} / "
    f"{unigrams_counts['Dy']} = {p_has_given_dy:.3f}"
)

# --- third step: calculate P(the|Dy, has) ---
count_dy_has_the = trigrams_counts[("Dy", "has", "the")]
count_dy_has_context = bigrams_counts[("Dy", "has")]
p_the_given_dy_has = count_dy_has_the / count_dy_has_context
print(
    f"third step: P(the|Dy, has) = {count_dy_has_the} / "
    f"{count_dy_has_context} = {p_the_given_dy_has:.3f}"
)

# --- fourth step: calculate P(wisdom|has, the) ---
count_has_the_wisdom = trigrams_counts[("has", "the", "wisdom")]
count_has_the_context = bigrams_counts[("has", "the")]
p_wisdom_given_has_the = count_has_the_wisdom / count_has_the_context
print(
    f"fourth step: P(wisdom|has, the) = {count_has_the_wisdom} / "
    f"{count_has_the_context} = {p_wisdom_given_has_the:.3f}"
)

# --- last step: calculate P('Dy has the wisdom') ---
p_sentence = (
    p_dy
    * p_has_given_dy
    * p_the_given_dy_has
    * p_wisdom_given_has_the
)
print(
    "last step: P('Dy has the wisdom') ≈ "
    f"{p_dy:.3f} * {p_has_given_dy:.3f} * "
    f"{p_the_given_dy_has:.3f} * {p_wisdom_given_has_the:.3f} "
    f"= {p_sentence:.3f}"
)
