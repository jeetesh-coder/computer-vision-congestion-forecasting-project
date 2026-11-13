# Original dataset

data/raw/Dataset contains 391 original train and 604 original test images,
with matching Supervisely JSON annotations. Total: 995 images.

The split manifest is data/splits/three_class_v1.json.
Three-class detection uses 312 train / 79 validation / 243 test images.
361 generic `vehicle` images are excluded from this detection experiment.
An unmatched annotation without an image is not copied.

Preserve original image names, sequence tags and target IDs.
These sampled images are not verified continuous video.
Check frame gaps and timing before tracking or forecasting.
Preserve upstream dataset attribution and redistribution terms.
