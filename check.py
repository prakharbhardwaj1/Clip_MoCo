import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional

import sys
sys.path.append('../')

from eval import evaluate, bootstrap
from zero_shot import make, make_true_labels, run_softmax_eval

## Define Zero Shot Labels and Templates

# ----- DIRECTORIES ------ #
cxr_filepath: str = '/home/woody/iwi5/iwi5190h/CheXzero/data/cxr_testnih.h5' # filepath of chest x-ray images (.h5)
cxr_true_labels_path: Optional[str] = '/home/woody/iwi5/iwi5190h/CheXzero/data/gt_test.csv' # (optional for evaluation) if labels are provided, provide path
model_path: str = '/home/woody/iwi5/iwi5190h/CheXzero/checkpoints/imagenet_run4/imagenet_run4.pt' # where pretrained models are saved (.pt) 
predictions_dir: Path = Path('/home/woody/iwi5/iwi5190h/CheXzero/predictions') # where to save predictions
cache_dir: str = predictions_dir / "imagenet_run4" # where to cache ensembled predictions

context_length: int = 77

# ------- LABELS ------  #
# Define labels to query each image | will return a prediction for each label
cxr_labels: List[str] = ["Atelectasis","Consolidation","Infiltration",
                        "Pneumothorax","Edema","Emphysema","Fibrosis",
                        "Effusion","Pneumonia","Pleural_Thickening",
                        "Cardiomegaly","Nodule","Mass","Hernia","No Finding"]

                        ####ttry diffrent names for the labels.

# ---- TEMPLATES ----- # 
# Define set of templates | see Figure 1 for more details                        
cxr_pair_template: Tuple[str] = ("{}", "no {}")

## Simplified model evaluation to use single checkpoint
def evaluate_model(
    model_path: str, 
    cxr_filepath: str, 
    cxr_labels: List[str], 
    cxr_pair_template: Tuple[str], 
    cache_dir: str = None, 
    save_name: str = None,
) -> np.ndarray:
    """
    Load model and return predictions. Caches predictions at `cache_dir` if location provided.
    """

    model_name = Path(model_path).stem
    
    # load in model and `torch.DataLoader`
    model, loader = make(
        model_path=model_path, 
        cxr_filepath=cxr_filepath, 
    ) 
    
    # path to the cached prediction
    cache_path = Path(cache_dir) / f"{save_name}_{model_name}.npy" if save_name else Path(cache_dir) / f"{model_name}.npy"

    # if prediction already cached, don't recompute prediction
    if os.path.exists(cache_path): 
        print("Loading cached prediction for {}".format(model_name))
        y_pred = np.load(cache_path)
    else: # cached prediction not found, compute preds
        print("Inferring model {}".format(model_path))
        y_pred = run_softmax_eval(model, loader, cxr_labels, cxr_pair_template)
        if cache_dir is not None: 
            Path(cache_dir).mkdir(exist_ok=True, parents=True)
            np.save(file=cache_path, arr=y_pred)
    
    return y_pred

# Run the model evaluation using the specified checkpoint
y_pred_avg = evaluate_model(
    model_path=model_path, 
    cxr_filepath=cxr_filepath, 
    cxr_labels=cxr_labels, 
    cxr_pair_template=cxr_pair_template, 
    cache_dir=cache_dir,
    save_name="imagenet_run4"
)

# Save the predictions
predictions_dir = predictions_dir / "imagenet_run4.npy"
np.save(file=predictions_dir, arr=y_pred_avg)

## If ground truth labels are available, compute AUC on each pathology to evaluate the performance of the zero-shot model.

# make test_true
test_true = make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)

# evaluate model
cxr_results = evaluate(y_pred_avg, test_true, cxr_labels)

# bootstrap evaluations for 95% confidence intervals
bootstrap_results = bootstrap(y_pred_avg, test_true, cxr_labels)


# display AUC with confidence intervals
df = pd.DataFrame(bootstrap_results[1])
pd.set_option('display.max_columns', None)  # None means no limit
pd.set_option('display.max_rows', None)  # Adjust if you also want to see all rows

# Calculate average AUC manually
avg_auc = np.mean(df.values, axis=1)

# Add average AUC column to the DataFrame
df['Average AUC'] = avg_auc
print(df)
# Saving the DataFrame to a CSV file
df.to_csv("/home/woody/iwi5/iwi5190h/CheXzero/predictions/AUC_imagenet_run4.csv")