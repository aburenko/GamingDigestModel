import json
from pathlib import Path

import click
import numpy as np


def stats(labels: np.ndarray, highlight_count: int) -> None:
    print(f"Number of highlight frames: {np.count_nonzero(labels)}")
    print(f"Number of non-highlight frames: {len(labels) - np.count_nonzero(labels)}")
    print(f"Rate of highlights: {np.count_nonzero(labels) / len(labels)}")
    print(f"Highlight count: {highlight_count}")


@click.command()
@click.argument('input_file')
def extract_labels(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    labels = []
    images = []
    seq_indeces = []

    is_highlight = False
    promise_to_end_highlight = False

    highlight_count = 0
    start_index = None
    for index, item in enumerate(data["frames"]):
        if promise_to_end_highlight:
            is_highlight = False
            promise_to_end_highlight = False
            seq_indeces.append((start_index, index))

        for label in item.get('labels', []):
            if label.get('category') == 'Highlight start':
                is_highlight = True
                highlight_count += 1
                start_index = index
            elif label.get('category') == 'Highlight end':
                promise_to_end_highlight = True

        images.append(item["name"])
        labels.append(1 if is_highlight else 0)

    labels = np.array(labels)
    stats(labels, highlight_count)
    print(seq_indeces)
    split(images, labels, seq_indeces)


def split(images, labels, indeces):
    # 10 last highlights are allocated for the test set.
    last_train_index = indeces[-11][1]
    save_images("test", images[last_train_index:], labels[last_train_index:])
    # We simplify train set to reduce number of not highlights in training data
    train_cut_indeces = np.zeros_like(labels, dtype=int)
    train_highlight_frame_count = 0
    for index_pair in indeces[:-10]:
        train_highlight_frame_count += index_pair[1] - index_pair[0]
        start_index = index_pair[0] - minute_in_5fps_with_deviation()
        end_index = index_pair[1] + minute_in_5fps_with_deviation()
        train_cut_indeces[start_index: end_index] = 1
    print(train_highlight_frame_count)
    selection = train_cut_indeces == True
    save_images("train", np.array(images)[selection], np.array(labels)[selection])


def minute_in_5fps_with_deviation():
    minute_in_5fps_frames = 300
    np.random.seed(42)
    return minute_in_5fps_frames + np.random.randint(-100, 300)


def save_images(phase, images, labels):
    Path(phase).mkdir(exist_ok=True)
    print(phase, "has", len(images), "instances")
    np.save(f"{phase}/labels.npy", labels)
    np.save(f"{phase}/images.npy", images)


if __name__ == '__main__':
    extract_labels()
