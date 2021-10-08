import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def make_submission(path: str, y_pred: np.ndarray) -> None:
    submission = pd.read_csv(
        os.path.join(path, "sample_submission.csv"), engine="python")
    submission["Pawpularity"] = y_pred
    submission.to_csv("submission.csv", index=False)


def plot_results(save_dir, name, train_results, val_results):
    plt.style.use('ggplot')
    plt.plot(train_results, label='training ' + name)
    plt.plot(val_results, label='validation ' + name)
    plt.legend()
    plt.savefig(os.path.join(save_dir, name + '.jpg'))
    plt.close()
