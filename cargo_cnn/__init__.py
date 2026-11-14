"""
cargo_cnn - a small convolutional network that classifies single cells as
released / not released, trained on your own blind labels.

    python -m cargo_cnn.train   --data "D:\\analysis\\Slide I"   (learn)
    python -m cargo_cnn.predict --model models\\<run>\\model.pt --parent "...\\pre_glucose" --samples all

Run both from the repository folder (the one holding cargo_measure.py), with
the same conda environment you use for Cellpose - it already contains torch.
See DOCUMENTATION.md, section 9, for how and why it works.
"""

CNN_VERSION = "1.0"
