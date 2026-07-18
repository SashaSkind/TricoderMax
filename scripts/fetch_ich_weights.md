# Getting ICH weights (reused, not trained)

The ICH module wraps a **public pretrained checkpoint** — we do not train from
scratch. Reference: the VinBigData RSNA-2019 Intracranial Hemorrhage CNN stage.

Place a checkpoint at:

    weights/ich/ich_cnn.pth

It must be a 2D CNN state_dict matching `TRICORDER_ICH_BACKBONE`
(default `tf_efficientnet_b0`, `in_chans=3`, 6 outputs = 5 subtypes + "any").
`ICHModel` unwraps `state_dict` / `model` containers and strips `module.` prefixes,
and loads with `strict=False` so a compatible head/backbone loads cleanly.

## Where these live
Winning RSNA-2019 solution weights are published as **Kaggle datasets** attached
to the authors' inference notebooks (not on GitHub). Fetch with the Kaggle API:

    pip install kaggle           # + ~/.kaggle/kaggle.json token
    # accept the competition rules once, then download the author's weights dataset
    kaggle datasets download -d <owner>/<ich-weights-dataset> -p weights/ich --unzip

Then adapt the state_dict to the backbone above if the author used a different one
(set `TRICORDER_ICH_BACKBONE` to match, e.g. `seresnext50_32x4d`).

Licensing: RSNA-2019 is non-commercial; reused weights inherit those terms.

## No weights yet? Wiring test only
    TRICORDER_ICH_ALLOW_RANDOM=1   # builds an UNTRAINED model

Results are then marked **uncalibrated** with a loud note and must never be
reported as real detection. This exists only to exercise the pipeline end-to-end.
