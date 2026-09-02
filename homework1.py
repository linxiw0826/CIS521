import numpy
import nltk
import string

############################################################
# CIS 521: Homework 1
############################################################

student_name = "Linxi Wu"

# This is where your grade report will be sent.
student_email = "linxiw@engineering.penn.edu"

############################################################
# Section 1: Python Concepts
############################################################

python_concepts_question_1 = (
    "Python is strongly typed because every object has a fixed type and "
    "values of "
    "incompatible types cannot be mixed directly without "
    "explicit conversion.\n"
    "For example, 2+3 raises a TypeError. Python is dynamically typed because "
    "variable names are bound to objects at runtime and "
    "the type of a variable can "
    "change over time.\n"
    "For example, assigning `x = 1` and then `x = 'hello'` is valid because "
    "`x` is just a name that can refer to different types."
)

python_concepts_question_2 = (
    "Dictionary keys in Python must be immutable. Lists are not immutable "
    "so they cannot be used as dictionary keys. A fix is to use immutable "
    "tuples for points, e.g. {"
    "(0, 0): 'home', (1, 2): 'school', (-1, 1): 'market'}."
)

python_concepts_question_3 = (
    "The `''.join(strings)` version is faster for large inputs because "
    "string concatenation with `+=` builds many temporary string objects "
    "(O(n^2) behavior in the aggregate) due to repeated copying). "
    "`join` precomputes the total size and builds the result in one pass."
)

############################################################
# Section 2: Working with Lists
############################################################


def extract_and_apply(lst, p, f):
    return [f(x) for x in lst if p(x)]


def concatenate(seqs):
    return [item for seq in seqs for item in seq]


def transpose(matrix):
    return [list(row) for row in zip(*matrix)]

############################################################
# Section 3: Sequence Slicing
############################################################


def copy(seq):
    return seq[:]


def all_but_last(seq):
    return seq[:-1]


def every_other(seq):
    return seq[::2]

############################################################
# Section 4: Combinatorial Algorithms
############################################################


def prefixes(seq):
    for i in range(len(seq) + 1):
        yield seq[:i]


def suffixes(seq):
    for i in range(len(seq)):
        yield seq[i:]
    yield seq[len(seq):]


def slices(seq):
    for i in range(len(seq)):
        for j in range(i + 1, len(seq) + 1):
            yield seq[i:j]

############################################################
# Section 5: Text Processing
############################################################


def normalize(text):
    return " ".join(text.lower().split())


def no_vowels(text):
    vowels = set("aeiouAEIOU")
    return "".join(ch for ch in text if ch not in vowels)


def digits_to_words(text):
    mapping = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    }
    words = [mapping[ch] for ch in text if ch.isdigit()]
    return " ".join(words)


def to_mixed_case(name):
    parts = name.split("_")
    if not parts:
        return ""
    first = parts[0].lower()
    rest = "".join(word.capitalize() for word in parts[1:] if word)
    return f"{first}{rest}"

############################################################
# Section 6: Polynomials
############################################################


class Polynomial(object):

    def __init__(self, polynomial):
        self.polynomial = tuple(polynomial)

    def get_polynomial(self):
        return self.polynomial

    def __neg__(self):
        return Polynomial(
            tuple((-coef, power) for (coef, power) in self.polynomial))

    def __add__(self, other):
        return Polynomial(self.polynomial + other.polynomial)

    def __sub__(self, other):
        return Polynomial(
            self.polynomial + tuple(
                (-coef, power) for (coef, power) in other.polynomial))

    def __mul__(self, other):
        return Polynomial((coef * other_coef, pwr + other_pwr)
                          for (coef, pwr) in self.polynomial
                          for (other_coef, other_pwr) in other.polynomial)

    def __call__(self, x):
        return sum(coef * (x ** power) for coef, power in self.polynomial)

    def simplify(self):
        terms = {}
        for coef, power in self.polynomial:
            terms[power] = terms.get(power, 0) + coef
        simplified = [
            (coef, power) for power, coef in terms.items() if coef != 0
        ]
        if not simplified:
            simplified = [(0, 0)]
        self.polynomial = tuple(
            sorted(
                simplified, key=lambda term: term[1], reverse=True
            )
        )

    def __str__(self):
        parts = []
        first = True
        for coef, power in self.polynomial:
            sign = "-" if coef < 0 else "+"
            magnitude = abs(coef)
            if power == 0:
                term = str(magnitude)
            elif magnitude == 1:
                term = "x" if power == 1 else f"x^{power}"
            elif power == 1:
                term = f"{magnitude}x"
            else:
                term = f"{magnitude}x^{power}"

            if first:
                if sign == "-":
                    parts.append(f"-{term}")
                else:
                    parts.append(term)
            else:
                parts.append(f" {sign} {term}")
            first = False
        return "".join(parts)

############################################################
# Section 7: Python Packages
############################################################


def sort_array(list_of_matrices):
    if not list_of_matrices:
        return numpy.array([], dtype=int)
    flattened = [numpy.array(matrix).reshape(-1)
                 for matrix in list_of_matrices]
    merged = numpy.concatenate(flattened)
    return numpy.sort(merged)[::-1]


def POS_tag(sentence):
    stop_words = set(nltk.corpus.stopwords.words("english"))
    words = nltk.word_tokenize(sentence.lower())
    words = [word for word in words
             if word not in stop_words and word not in string.punctuation]
    return nltk.pos_tag(words)

############################################################
# Section 8: Feedback
############################################################


# Just an approximation is fine.
feedback_question_1 = """
3 Hours
"""

feedback_question_2 = """
Section7
"""

feedback_question_3 = """
Section5
"""
