import argparse
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

from pathlib import Path
from pos_tagger.predict import Predictor
from pos_tagger.train import Trainer
from features.feature_extractor import FeatureExtractor
from utils.dataset_creator import DatasetCreator
from features.features import Features


class MEMM(object):
    def __init__(self, opts: argparse.ArgumentParser, dir: Path):

        super(MEMM, self).__init__()

        self.dir = dir

        self.corpus = FeatureExtractor.process(opts.train_file, opts.force)
        self.corpus.filter(**opts.features_params)

        self.features = Features()
        self.features.from_data(self.corpus.dicts)

        self.ds_tags = list(self.corpus.dicts["unigrams"].keys())
        self.ds_tags_dict = {i: k for i, k in enumerate(self.ds_tags)}

        self.train_file = opts.train_file
        self.ds_name = self.train_file.stem
        self.ds_index = int(self.ds_name[-1])

        self.post_processing = opts.post_processing

    def fit(self, opts: argparse.ArgumentParser):
        dataset = DatasetCreator(self.train_file, self.features, self.ds_tags)
        true_features = dataset.to_features()
        list_features = dataset.process()
        trainer = Trainer(true_features, list_features, reg_lambda=opts.reg_lambda, num_features=len(self.features))
        optimal_params = trainer.optimize(epochs=opts.epochs, print_every=opts.print_every, eps=opts.epsilon)
        self.weights = optimal_params[0]

        w_path = self.dir.joinpath(f"trained_weights_data_{self.ds_index}.pkl")
        with open(w_path, "wb") as f:
            pickle.dump(optimal_params, f)

    def predict(self, test_file, beam: int = 2):

        predictor = Predictor(self.weights, self.features, self.ds_tags_dict, beam=beam)

        predictions = []
        num_lines = sum(1 for _ in open(test_file))
        with open(test_file, "r") as f:

            for i, line in enumerate(f.readlines()):

                pairs = [tuple(s.split("_")) for s in line.split()]

                sentence = list(list(zip(*pairs))[0])
                real_tags = list(list(zip(*pairs))[1])

                preds = predictor.predict(sentence, self.post_processing)

                pred_line = " ".join([w + "_" + t for w, t in zip(sentence, preds)])
                predictions.append(pred_line)
                predictor.append_stats(real_tags, preds)

                print(
                    f"\rPREDICTING  |  CORPUS = `{test_file.name}` [{round(100. * (i + 1) / num_lines, 2)}%%]", end=""
                )
            print()

        accuracy, confusion = predictor.get_stats()
        print(f"\rPREDICTING  |  CORPUS = `{test_file.name}` |  ACCURACY {accuracy :.2f}%")

        w_path = self.dir.joinpath(f"{test_file.stem}.wtag")
        with open(w_path, "w") as f:
            f.writelines("\n".join(predictions))

        w_path = self.dir.joinpath(f"{test_file.stem}_confusion.pdf")
        plt.figure(figsize=(15, 15))
        sns.heatmap(
            confusion, annot=False, cmap="Blues", xticklabels=self.ds_tags, yticklabels=self.ds_tags,
        )
        plt.tight_layout()
        plt.savefig(w_path)
        plt.close()

        return predictions, accuracy

    def predict_no_gt(self, test_file, beam: int = 2):
        predictor = Predictor(self.weights, self.features, self.ds_tags_dict, beam=beam)

        predictions = []
        num_lines = sum(1 for _ in open(test_file))
        with open(test_file, "r") as f:

            for i, line in enumerate(f.readlines()):

                pairs = [tuple(s.split("_")) for s in line.split()]
                sentence = list(list(zip(*pairs))[0])
                preds = predictor.predict(sentence, self.post_processing)

                pred_line = " ".join([w + "_" + t for w, t in zip(sentence, preds)])
                predictions.append(pred_line)

                print(
                    f"\rPREDICTING  |  CORPUS = `{test_file.name}` [{round(100. * (i + 1) / num_lines, 2)}%%]", end=""
                )
            print()

        w_path = self.dir.joinpath(f"{test_file.stem}.wtag")
        with open(w_path, "w") as f:
            f.writelines("\n".join(predictions))
