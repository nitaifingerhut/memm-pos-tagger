import abc
import sys

from utils.history import History
from utils.general import is_digit
import inspect


#######################################################################
class Feature(object):
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def __call__(self, history: History) -> int:
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


#######################################################################


#######################################################################
class PairFeature(object):
    def __init__(self, word, tag):
        self.word = word
        self.tag = tag

    @abc.abstractmethod
    def __call__(self, history: History) -> int:
        return history.words[history.index] == self.word and history.tags[-1] == self.tag

    def __str__(self):
        pair_str = f"word = {self.word}, tag = {self.tag}"
        return self.__class__.__name__ + ": " + pair_str


#######################################################################


#######################################################################
class NGramFeature(Feature):
    def __init__(self, tags):
        self.tags = tags

    def __call__(self, history: History):
        raise NotImplementedError

    def __str__(self):
        tags_str = ", ".join(self.tags)
        return self.__class__.__name__ + ": " + tags_str


class UniGramFeature(NGramFeature):
    def __call__(self, history: History):
        return history.tags[-1] == self.tags[-1]


class BiGramFeature(NGramFeature):
    def __call__(self, history: History):
        return history.tags[-2:] == self.tags[-2:]


class TriGramFeature(NGramFeature):
    def __call__(self, history: History):
        return history.tags[-3:] == self.tags[-3:]


#######################################################################

#######################################################################
class PreSufFeature(Feature):
    def __init__(self, chars):
        self.chars = chars
        self.n = len(chars)

    def __call__(self, history: History):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__ + ": " + self.chars


class PrefixFeature(PreSufFeature):
    def __call__(self, history: History):
        w = history.words[history.index]
        return len(w) >= 6 and w[: self.n] == self.chars


class SuffixFeature(PreSufFeature):
    def __call__(self, history: History):
        w = history.words[history.index]
        return len(w) >= 6 and w[-self.n :] == self.chars


#######################################################################


#######################################################################

class IndexFeature(Feature):
    def __init__(self, prams):
        self.check = prams[0]
        self.inx = prams[1]

class IndexTagFeature(Feature):
    def __call__(self, history: History) -> int:
        w = history.words[history.index]
        return history.tags[-1] == self.check and history.index == self.inx

class IndexWordFeature(Feature):
    def __call__(self, history: History) -> int:
        w = history.words[history.index]
        return history.words[history.index] == self.check and history.index == self.inx

#######################################################################


#######################################################################
class Custom001(Feature):
    def __call__(self, history: History) -> int:
        w = history.words[history.index]
        return w.upper()[0] == w[0]

    def __str__(self):
        return f"Feature: current word start with a capital letter"


class Custom002(Feature):
    def __call__(self, history: History) -> int:
        w = history.words[history.index]
        return w.upper() == w

    def __str__(self):
        return f"Feature: current word is capitalized"


class Custom003(Feature):
    def __call__(self, history: History) -> int:
        w = history.words[history.index]
        return is_digit(w)

    def __str__(self):
        return f"Feature: current word is a digit"


class Custom004(Feature):
    def __call__(self, history: History) -> int:
        return history.index == 0

    def __str__(self):
        return f"Feature: current word is first in line"


class Custom005(Feature):
    def __call__(self, history: History) -> int:
        return history.index == 1

    def __str__(self):
        return f"Feature: current word is second in line"


class Custom006(Feature):
    def __call__(self, history: History) -> int:
        return history.index == 2

    def __str__(self):
        return f"Feature: current word is third in line"


#######################################################################


#######################################################################
FEATURES_DICT = {
    name: obj() for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass) if name.startswith("Custom")
}
#######################################################################
