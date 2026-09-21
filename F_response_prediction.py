# ------------------------------------------------------------
# Predict treatment response from baseline hctsa features
# ------------------------------------------------------------
# Run MATLAB script E first. It selects features using the training
# participants only and saves their raw baseline values for each fold.
#
# This script uses Python/scikit-learn to fit the response SVM. It does
# not calculate hctsa features or repeat the feature-selection step.
# Each fold trains on 41 participants and predicts the remaining person.
# The model uses raw feature values, without additional scaling.
#
# Run from the top. Results are written to a new folder at the end;
# this short script does not save checkpoints for pausing/resuming.


# %% 1. Import the packages used below

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)


# %% 2. File paths — change these if running on another computer

# MATLAB script E saves the selected baseline features here.
preparation_folder = Path(
    "/Users/aron/Desktop/Neural_Networks_Paper_New_Location/GitHub_Manuscript_Code/outputs/response_features"
)

# New results go here. An existing folder will not be overwritten.
output_folder = Path(
    "/Users/aron/Desktop/Neural_Networks_Paper_New_Location/GitHub_Manuscript_Code/outputs/response_results"
)

# This workbook supplies participant IDs and responder labels only.
# Its historical feature columns are not used as model inputs.
data_file = Path(
    "/Users/aron/Desktop/Neural_Networks_Paper_New_Location/from_google_drive/Results/Data_4_exploratory_SVM/Selected_top_features_for_SVM.xlsx"
)


# %% 3. Set the number of folds, permutations and random seeds

number_of_folds = 42       # Leave one participant out at a time.
n_perm = 1000              # Repeat with shuffled responder labels.
seed = 0                  # Fixed seed for the SVM.
permutation_seed = 1       # Fixed seed for the label shuffles.

assert not output_folder.exists(), "Choose a fresh output folder."

# Stop if the SVM reports a convergence problem, rather than using that fit.
warnings.filterwarnings("error", category=ConvergenceWarning)


# %% 4. Read the clinical labels and participant folds

# The workbook and MATLAB participant list must describe the same people
# in the same order. The checks below confirm the expected study ordering.
df = pd.read_excel(data_file)
participants = pd.read_csv(preparation_folder / "participant_folds.csv")

# Convert the text labels to numbers: 0 = nonresponder, 1 = responder.
labels_raw = df["Responder"].astype(str).str.strip().str.lower()
label_map = {"nonresponder": 0, "responder": 1}
y = labels_raw.map(label_map).astype(int).values

# Fold 1 holds out participant 1, fold 2 holds out participant 2, and so on.
fold_assignment = participants["Fold"].to_numpy(dtype=int)


# %% 5. Load the baseline features selected by MATLAB

# Each fold has its own feature set, selected without its test participant.
# X_by_fold stores one matrix per fold: participants in rows, features in
# columns. The feature_counts list records how many came from each PC.
X_by_fold = {}
feature_counts = []

for fold in range(1, number_of_folds + 1):

    # Collect this fold's selected features from PCs 1, 2 and 3.
    blocks = []

    for pc in range(1, 4):
        path = preparation_folder / f"fold{fold:02d}_PC{pc}.mat"
        saved = loadmat(path, squeeze_me=True)

        # Keep all 42 baseline rows; the training/test split is applied later.
        X = np.asarray(saved["rawBaseline"], dtype=float).reshape(42, -1)
        assert np.isfinite(X).all(), "Check invalid baseline values before fitting."

        # MATLAB's training mask covers baseline AND post recordings.
        # Confirm that both recordings from the test person were excluded.
        assert np.array_equal(
            saved["isTraining"].astype(bool),
            np.tile(fold_assignment != fold, 2),
        )

        blocks.append(X)
        feature_counts.append(
            dict(Fold=fold, PC=pc, NumberFeatures=X.shape[1])
        )

    # Place the three PCs' selected features alongside each other.
    X_by_fold[fold] = np.column_stack(blocks)


# %% 6. Check participant identities and the response groups

# Subject numbers are reused between treatments, so check the treatment
# blocks as well as the numbers. Do not silently reorder either input.
assert len(participants) == len(df) == len(y) == 42
assert participants.Treatment.tolist() == ["ECT"] * 21 + ["MST"] * 21
assert np.array_equal(
    df.Subject.to_numpy(),
    participants.Participant.str[3:].astype(int).to_numpy(),
)

# The study has 21 responders and 21 nonresponders, with 42 LOPO folds.
assert np.array_equal(np.bincount(y), [21, 21])
assert np.array_equal(fold_assignment, np.arange(1, 43))


# %% 7. Prepare the real labels and 1,000 shuffled versions

# Row 0 contains the true labels. The remaining rows contain permutations.
# Shuffling changes which people carry each label, preserving group sizes.
# Every fold uses the same label arrangement for a given permutation.
rng = np.random.default_rng(permutation_seed)
labels = np.vstack([y] + [rng.permutation(y) for _ in range(n_perm)])

# Store predictions for all participants in every real/shuffled analysis.
# -1 and NaN mark entries that have not yet been filled.
y_pred = np.full(labels.shape, -1, dtype=int)
y_dec = np.full(42, np.nan)


# %% 8. Fit the SVM and predict each held-out participant

# First use the true labels (permutation 0), then each shuffled arrangement.
# Feature selection is unchanged: it used pre/post labels, not response labels.
for permutation in range(n_perm + 1):

    for fold in range(1, number_of_folds + 1):

        # Train on the other 41 people and test on this fold's one person.
        train = fold_assignment != fold
        test = fold_assignment == fold
        X = X_by_fold[fold]
        y_fit = labels[permutation]

        # Fit the linear SVM to the selected raw baseline values.
        clf = SVC(
            kernel="linear",
            C=1.0,
            probability=True,
            random_state=seed,
        )
        clf.fit(X[train], y_fit[train])
        assert clf.fit_status_ == 0, "SVM did not converge; investigate before using."

        # Save the predicted class for the held-out person: 0 or 1.
        y_pred[permutation, test] = clf.predict(X[test])

        # For the real-label analysis, also save the continuous decision score
        # for calculating AUC. These scores are not predicted probabilities.
        if permutation == 0:
            assert np.array_equal(clf.classes_, [0, 1])
            y_dec[test] = clf.decision_function(X[test])

    if permutation % 50 == 0:
        print(f"Completed permutation {permutation}/{n_perm}.", flush=True)


# %% 9. Calculate performance using the real-label predictions

# Check that every prediction and observed decision score was filled.
assert np.isin(y_pred, [0, 1]).all()
assert np.isfinite(y_dec).all()

# Rows are actual classes; columns are predicted classes.
# tn/fp: correctly/incorrectly classified nonresponders.
# fn/tp: incorrectly/correctly classified responders.
cm = confusion_matrix(y, y_pred[0], labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

acc = accuracy_score(y, y_pred[0])
bal_acc = balanced_accuracy_score(y, y_pred[0])

# Sensitivity: proportion of responders correctly identified.
# Specificity: proportion of nonresponders correctly identified.
sens = tp / (tp + fn)
spec = tn / (tn + fp)

# AUC describes how well the decision scores rank responders above
# nonresponders, across possible classification thresholds.
auc = roc_auc_score(y, y_dec)


# %% 10. Compare observed performance with the shuffled-label results

# Compare each set of predictions with the labels used for that analysis.
# There are 21 people per class in every permutation, so pooled accuracy
# equals balanced accuracy. Integer counts avoid rounding when checking ties.
correct_counts = (y_pred == labels).sum(axis=1)
assert np.isclose(bal_acc, correct_counts[0] / 42)
perm_bal = correct_counts[1:] / 42

# Count shuffled results at least as good as the observed result.
# Add one to both counts when calculating the permutation p-value.
exceedances = int(np.sum(correct_counts[1:] >= correct_counts[0]))
p_perm = (1 + exceedances) / (n_perm + 1)

print("Confusion matrix (rows=true [0,1], columns=predicted [0,1]):")
print(cm)
print(f"Balanced accuracy = {bal_acc:.4f}; AUC = {auc:.4f}; p = {p_perm:.5f}")


# %% 11. Save the summary and predictions

# Store the main results in a named summary, including the number of selected
# features in each fold and the mean/SD of the shuffled accuracies.
results = dict(
    analysis="LOPO treatment-response prediction, raw baseline inputs",
    n_participants=42,
    number_features_by_fold=[
        X_by_fold[k].shape[1] for k in range(1, number_of_folds + 1)
    ],
    correct=int(correct_counts[0]),
    accuracy=float(acc),
    balanced_accuracy=float(bal_acc),
    sensitivity=float(sens),
    specificity=float(spec),
    auc=float(auc),
    confusion_matrix=cm.tolist(),
    permutation_exceedances=exceedances,
    permutation_p=p_perm,
    null_mean=float(perm_bal.mean()),
    null_sd_sample=float(perm_bal.std(ddof=1)),
    number_of_permutations=n_perm,
)

output_folder.mkdir(parents=True)

# Overall performance summary.
(output_folder / "metrics.json").write_text(
    json.dumps(results, indent=2) + "\n"
)

# One row per participant, with their real label and held-out prediction.
# This file contains participant information and must remain private.
participants.assign(
    Response=y,
    PredictedResponse=y_pred[0],
    DecisionScore=y_dec,
).to_csv(output_folder / "held_out_predictions.csv", index=False)

# One row per permutation, showing its number correct and balanced accuracy.
pd.DataFrame(
    dict(
        Permutation=np.arange(1, n_perm + 1),
        Correct=correct_counts[1:],
        BalancedAccuracy=perm_bal,
    )
).to_csv(output_folder / "permutation_results.csv", index=False)

# Keep the complete prediction arrays and label schedule for later checking.
# This file also contains participant-level results and must remain private.
np.savez_compressed(
    output_folder / "predictions.npz",
    predictions=y_pred,
    observed_scores=y_dec,
    labels=labels,
)
