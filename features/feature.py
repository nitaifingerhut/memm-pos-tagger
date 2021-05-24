import abc
import sys

from utils.history import History
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
        pair_str = f"word={self.word}, tag={self.tag}"
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
        return len(w) >= self.n and w[: self.n] == self.chars


class SuffixFeature(PreSufFeature):
    def __call__(self, history: History):
        w = history.words[history.index]
        return len(w) >= self.n and w[-self.n :] == self.chars


#######################################################################


#######################################################################
class f101(Feature):
    def __call__(self, history: History) -> int:
        w = history.words[history.index]
        return w.endswith("ing") and history.tags[-1] == "VBG"


class f102(Feature):
    def __call__(self, history: History) -> int:
        w = history.words[history.index]
        return w.startswith("pre") and history.tags[-1] == "NN"


class f103(Feature):
    def __call__(self, history: History) -> int:
        try:
            return history.tags[-3:] == ("DT", "JJ", "Vt")
        except IndexError:
            return 0


class f104(Feature):
    def __call__(self, history: History) -> int:
        try:
            return history.tags[-2:] == ("JJ", "Vt")
        except IndexError:
            return 0


class f105(Feature):
    def __call__(self, history: History) -> int:
        return history.tags[-1] == "Vt"


class f106(Feature):
    def __call__(self, history: History) -> int:
        try:
            return 1 if history.words[history.index - 1] == "the" and history.tags[-1] == "Vt" else 0
        except IndexError:
            return 0


#######################################################################


#######################################################################
class n001(Feature):
    def __call__(self, history: History) -> int:
        w = history.words[history.index]
        return 1 if w.upper()[0] == w[0] else 0

    def __str__(self):
        return f"Feature: Current Word Starts With A Capital Letter"


class n002(Feature):
    def __call__(self, history: History) -> int:
        w = history.words[history.index]
        return 1 if w.upper() == w else 0

    def __str__(self):
        return f"Feature: Current Word Consist Only Of Capital Letters"


class n003(Feature):
    def __call__(self, history: History) -> int:
        w = history.words[history.index]
        return 1 if w.isdigit() else 0

    def __str__(self):
        return f"Feature: Current Word Is A Digit"


#######################################################################


#######################################################################
# FEATURES_DICT = {name: obj() for name, obj in
#                  inspect.getmembers(sys.modules[__name__], inspect.isclass)
#                  if obj.__module__ is __name__}
# del FEATURES_DICT["Features"]
#######################################################################
