
## Dependencies
To clone all files:

```git clone https://github.com/rajpurkarlab/CheXzero.git```

To install Python dependencies:

```pip install -r requirements.txt```


## Running Training
Run the following command to perform CheXzero pretraining. 
```bash
python run_train.py --cxr_filepath "./data/cxr.h5" --txt_filepath "data/mimic_impressions.csv"
```

## Citation
```bash
Tiu, E., Talius, E., Patel, P. et al. Expert-level detection of pathologies from unannotated chest X-ray images via self-supervised learning. Nat. Biomed. Eng (2022). https://doi.org/10.1038/s41551-022-00936-9
```

## License
The source code for the site is licensed under the MIT license, which you can find in the `LICENSE` file. Also see `NOTICE.md` for attributions to third-party sources. 
