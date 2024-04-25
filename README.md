# GamingDigestModel
Create digest from an unfiltered stream recoding on example of DayZ Streams.

## Example Steps
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
- run main.py

## Results
I labeled highlight sequences for a 4.5h stream. Resulting labels can be found in data/. We deal with highly unbalanced data:

Number of highlight frames: 4371
Number of non-highlight frames: 79565
Rate of highlights: 0.052
Highlight count: 42




## Initial planning
Plan:
- add download youtube video script: https://github.com/pytube/pytube (1h)
- Label data: 2 min blocks with highlight 1 or not 0. (2-3h)
   - ".yaml" with timestamps format: 0:0, 0:2, ... 0:30, ... 1:0, 1:2 ..., where 1:2 is video part 1h02m to 1ht04m   
- preprocess two streams: audio and image with (4h)
  - mean sound power/s
  - (1 scaled image)/s
  - train-test-split: half-half depending on highlight dist.
- model: ResNet50 for images with MLP head (including mean sound) (2h) https://pytorch.org/vision/0.17/models/video_resnet.html
  - loss function: binary cross entropy function
  - eval metric: F1, Recall, Precision. Number of positive samples is low -> Recall is important. FPs are not bad.
 