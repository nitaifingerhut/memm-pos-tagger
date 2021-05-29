import argparse

from datetime import datetime
from pathlib import Path


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = int(value)


class Parser(object):
    @staticmethod
    def parse():
        """
        Parse command-line arguments
        :return: argparser object with user opts.
        """
        parser = argparse.ArgumentParser()

        parser.add_argument("--train-file", type=Path, required=True, help="path to train file")
        parser.add_argument("--force", default=False, action="store_true", help="force processing train file")
        parser.add_argument("--test-file", type=Path, required=True, help="path to train file")
        parser.add_argument(
            "--name", type=str, default=datetime.now().strftime("%m-%d_%H-%M-%S"), help="name",
        )
        parser.add_argument(
            "--features-params",
            nargs="*",
            action=ParseKwargs,
            default={
                "pairs": 50,
                "unigrams": 25,
                "bigrams": 50,
                "trigrams": 100,
                "prefixes": 25,
                "suffixes": 25,
                "prev_w_curr_t": 25,
                "next_w_curr_t": 25,
                "index_tag": 25,
                "index_word": 25,
            },
        )
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--print-every", type=int, default=50)
        parser.add_argument("--reg-lambda", type=float, default=0.01)

        parser.add_argument("--beam", type=int, default=2, help="beam search width")
        parser.add_argument(
            "--post-processing", default=False, action="store_true", help="set to apply post processing"
        )

        opt = parser.parse_args()

        return opt
