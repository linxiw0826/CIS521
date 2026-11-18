############################################################
# CIS 521: Homework 9
############################################################

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import homework9_data as data

############################################################

student_name = "Linxi Wu"

############################################################
# Section 1: Perceptrons
############################################################


def _dot(w, x):
    return sum(w.get(k, 0) * v for k, v in x.items())


def _add(w, x, scale):
    for k, v in x.items():
        w[k] = w.get(k, 0) + scale * v


class BinaryPerceptron(object):

    def __init__(self, examples, iterations):
        self.w = {}
        for _ in range(iterations):
            for x, y in examples:
                if (_dot(self.w, x) > 0) != y:
                    _add(self.w, x, 1 if y else -1)

    def predict(self, x):
        return _dot(self.w, x) > 0


class MulticlassPerceptron(object):

    def __init__(self, examples, iterations):
        self.labels = []
        for _, y in examples:
            if y not in self.labels:
                self.labels.append(y)
        self.w = {y: {} for y in self.labels}
        for _ in range(iterations):
            for x, y in examples:
                guess = self.predict(x)
                if guess != y:
                    _add(self.w[y], x, 1)
                    _add(self.w[guess], x, -1)

    def predict(self, x):
        return max(self.labels, key=lambda lab: _dot(self.w[lab], x))

############################################################
# Section 2: Applications
############################################################


def _vector(instance):
    return {i: v for i, v in enumerate(instance)}


class IrisClassifier(object):

    def __init__(self, data):
        self.p = MulticlassPerceptron(
            [(_vector(x), y) for x, y in data], 20)

    def classify(self, instance):
        return self.p.predict(_vector(instance))


class DigitClassifier(object):

    def __init__(self, data):
        self.p = MulticlassPerceptron(
            [(self.feat(x), y) for x, y in data], 40)

    @staticmethod
    def feat(instance):
        f = {i: v / 16 for i, v in enumerate(instance) if v}
        f["bias"] = 1
        return f

    def classify(self, instance):
        return self.p.predict(self.feat(instance))


class BiasClassifier(object):

    def __init__(self, data):
        self.p = BinaryPerceptron(
            [(self.feat(x), y) for x, y in data], 10)

    @staticmethod
    def feat(x):
        return {"x": x, "bias": 1}

    def classify(self, instance):
        return self.p.predict(self.feat(instance))


class MysteryClassifier1(object):

    def __init__(self, data):
        self.p = BinaryPerceptron(
            [(self.feat(x), y) for x, y in data], 100)

    @staticmethod
    def feat(x):
        return {"r2": x[0] ** 2 + x[1] ** 2, "bias": 1}

    def classify(self, instance):
        return self.p.predict(self.feat(instance))


class MysteryClassifier2(object):

    def __init__(self, data):
        self.p = BinaryPerceptron(
            [(self.feat(x), y) for x, y in data], 20)

    @staticmethod
    def feat(x):
        return {"xyz": x[0] * x[1] * x[2], "bias": 1}

    def classify(self, instance):
        return self.p.predict(self.feat(instance))

############################################################
# Section 3: Feedback
############################################################


# Just an approximation is fine.
feedback_question_1 = """
3
"""

feedback_question_2 = """
The most challenging part was finding the right feature transformations for
the mystery data sets, and realizing the perceptron needs more iterations when
the features are not well scaled.
"""

feedback_question_3 = """
I liked seeing how a simple perceptron becomes powerful once the right
features are added. Plotting the data first made the transformations clear.
"""
