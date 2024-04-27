# GamingDigestModel (Proof of Concept)

Finds highlights in a game video stream recoding on example of DayZ Streams using ResNet18 and LSTM.

Please note: this project is just a Proof of Concept. Thus, the values are hardcoded and no configs are provided.

## Idea

Once I have heard streamer say that he would give a lot for a model that compiles
raw streams into video with highlights. This project adresses the issue and creates a MWP
for such a problem.

Example of integration: App which takes a stream as an input and provides interface for quickly
creating compilation videos for a YouTube channel. Such a service can potentially be sold for a small fee and
provide a great use for a streamer.

## Pipeline

We take an example video: https://www.youtube.com/watch?v=kt8wMx_c22g

- use prepare_video.py script to download the video.

```bash 
python prepare_video.py https://www.youtube.com/watch?v=kt8wMx_c22g br-stream
```

- use scalable for labeling (https://github.com/scalabel/scalabel/tree/master)
    - prepare video e.g:
  ```bash
   python -m scalabel.tools.prepare_data -i ~/PycharmProjects/GamingDigestModel/data/br-stream.mp4 -o ./br-stream --fps 5 --url-root http://localhost:8686/items/br-stream 
  ```
    - label (project setup is shown in misc/settings.png) and save the labels into data
- process labels with **data/extract_labels.py**
- run **main.py** to train or submit a job to slurm with **slurm_run.sh**

## Results

I labeled highlight sequences for a 4.5h stream. Resulting labels can be found in data/. The data is highly unbalanced:

- Number of highlight frames: 4371
- Number of non-highlight frames: 79565
- Rate of highlights: 0.052
- Highlight count: 42

Therefore the not highlight parts in the train set are reduced to be more balanced
(see data/extract_labels.py) resulting into:

- test with 17968 frames
- train with 16045 frames

The first attempt best results for traind CNN+LSTN model are shown here:

| Dataset | Precision | Recall           |
|---------|-----------|------------------|
| TRAIN   | 0.910     | 1153/3403 = 0.34 |
| TEST    | 0.024     | 222/968 = 0.23   |

It seems like the model predicts all samples as highlight, so L2-norm was tried to prevent overfitting.
This made it for the model difficult to converge and therefore dropout was introduced to LSTM. This also didn't lead
to a significant improvement. Finally, we unfreeze the ResNet and try again. The network fails to generalize again.

Further splitting frames into blocks can simplify the task a lot and will address the initial streamer-platform idea
much better. Changing the approach shows significant benefit:

| Dataset | Precision | Recall      |
|---------|-----------|-------------|
| TRAIN   | 1.0       | 48/48 = 1.0 |
| TEST    | 0.125     | 6/14 = 0.43 |

### Conclusion and Future work

It looks like the sample size is still too small to properly generalize with using barely the 4h video.
The labeling is costly.
As a solution an automatic data labeling could be developed, e.g. find stream/highlight videos pairs and create
automatic matching.

Moreover, the sound is ignored for now. This could bring a lot of improvement since gunshots is often a signifier for
a highlight!

As for model using VisTR should be considered as it constantly shows better performance in SOTA methods.

## Log (Initial planning)

Plan:

- add download youtube video script: https://github.com/pytube/pytube (1h)
- Label data: 2 min blocks with highlight 1 or not 0. (2-3h)
    - ".yaml" with timestamps format: 0:0, 0:2, ... 0:30, ... 1:0, 1:2 ..., where 1:2 is video part 1h02m to 1ht04m
- preprocess two streams: audio and image with (4h)
    - mean sound power/s
    - (1 scaled image)/s
    - train-test-split: half-half depending on highlight dist.
- model: ResNet50+LSTM for images with MLP head (including mean sound) (
  2h) https://pytorch.org/vision/0.17/models/video_resnet.html
    - loss function: binary cross entropy function
    - eval metric: F1, Recall, Precision. Number of positive samples is low -> Recall is important. FPs are not bad.
 