# GamingDigestModel
Create digest from an unfiltered stream recoding 

Plan:
- add download youtube video script: https://github.com/pytube/pytube (1h)
- Label data: 2 min blocks with highlight 1 or not 0. (2-3h)
   - ".yaml" with timestamps format: 0:0, 0:2, ... 0:30, ... 1:0, 1:2 ..., where 1:2 is video part 1h02m to 1ht04m   
- preprocess two streams: audio and image with (4h)
  - mean sound power/s
  - (1 scaled image)/s
  - train-test-split: half-half depending on highlight dist.
- model: ResNet50 for images with MLP head (including mean sound) (2h)
  - loss function: binary cross entropy function
  - eval metric: F1, Recall, Precision
 
