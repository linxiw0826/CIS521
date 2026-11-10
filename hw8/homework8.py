############################################################
# CIS 521: Homework 8
############################################################

############################################################
# Imports
############################################################

import math
import random
import string

############################################################

student_name = "Linxi Wu"

############################################################
# Section 1: Ngram Models
############################################################


def tokenize(text):
    tokens = []
    current = []
    for ch in text:
        if ch.isspace():
            if current:
                tokens.append("".join(current))
                current = []
        elif ch in string.punctuation:
            if current:
                tokens.append("".join(current))
                current = []
            tokens.append(ch)
        else:
            current.append(ch)
    if current:
        tokens.append("".join(current))
    return tokens


def ngrams(n, tokens):
    padded = ["<START>"] * (n - 1) + list(tokens) + ["<END>"]
    return [(tuple(padded[i:i + n - 1]), padded[i + n - 1])
            for i in range(len(padded) - n + 1)]


class NgramModel(object):

    def __init__(self, n):
        self.n = n
        self.counts = {}
        self.totals = {}

    def update(self, sentence):
        for context, token in ngrams(self.n, tokenize(sentence)):
            token_counts = self.counts.setdefault(context, {})
            token_counts[token] = token_counts.get(token, 0) + 1
            self.totals[context] = self.totals.get(context, 0) + 1

    def prob(self, context, token):
        total = self.totals.get(context, 0)
        if total == 0:
            return 0.0
        return self.counts[context].get(token, 0) / total

    def random_token(self, context):
        r = random.random()
        total = 0.0
        tokens = sorted(self.counts[context])
        for token in tokens:
            total += self.prob(context, token)
            if r < total:
                return token
        return tokens[-1]

    def random_text(self, token_count):
        start = ("<START>",) * (self.n - 1)
        context = start
        result = []
        for _ in range(token_count):
            token = self.random_token(context)
            result.append(token)
            if token == "<END>":
                context = start
            elif self.n > 1:
                context = context[1:] + (token,)
        return " ".join(result)

    def perplexity(self, sentence):
        grams = ngrams(self.n, tokenize(sentence))
        log_sum = 0.0
        for context, token in grams:
            p = self.prob(context, token)
            if p == 0:
                return float("inf")
            log_sum -= math.log(p)
        return math.exp(log_sum / len(grams))


def create_ngram_model(n, path):
    model = NgramModel(n)
    with open(path) as f:
        for line in f:
            model.update(line)
    return model

############################################################
# Section 2: Feedback
############################################################


# Just an approximation is fine.
feedback_question_1 = """
3
"""

feedback_question_2 = """
The most challenging part was getting the n-gram padding and the context
updates right, especially the n = 1 case where the context is always empty.
"""

feedback_question_3 = """
I liked seeing the generated text become more coherent as n increases. It
would be nice to also see smoothing for unseen n-grams.
"""
