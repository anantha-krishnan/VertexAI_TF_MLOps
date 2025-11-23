from jigsaw_classifier_updates import JigsawClassifier
from cfg import CFG

if __name__ == "__main__":
    classifier = JigsawClassifier(CFG)
    classifier.k_fold_model_training()
    #y_pred = classifier.predict()
    classifier.plot_metrics()