# Molecular ADMET & Drug-Likeness Screener

A web app that evaluates whether a molecule has the properties of a viable drug. Enter a molecule by name, pick one from a built-in library, or paste a SMILES string, and the app screens it across the pharmaceutical **ADMET framework** (Absorption, Distribution, Metabolism, Excretion, Toxicity), showing the structure, identity, and a breakdown of results.

It combines trained machine learning models with rule-based cheminformatics checks, and is deliberate about stating what it can and cannot do.

**Live app:** https://predictor-v2.streamlit.app/

**Built with:** Python, RDKit, scikit-learn, Streamlit, PyTDC

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [The ADMET Framework](#the-admet-framework)
- [Features in Detail](#features-in-detail)
  - [Absorption and Drug-Likeness](#absorption-and-drug-likeness)
  - [Distribution](#distribution)
  - [Metabolism](#metabolism)
  - [Excretion](#excretion)
  - [Toxicity and Structural Alerts](#toxicity-and-structural-alerts)
  - [Synthesizability](#synthesizability)
- [The Machine Learning Models](#the-machine-learning-models)
- [The Chemistry](#the-chemistry)
- [Project Structure](#project-structure)
- [How to Run It](#how-to-run-it)
- [Limitations and Honest Caveats](#limitations-and-honest-caveats)
- [Possible Improvements](#possible-improvements)

---

## What This Project Does

Most candidate molecules never become drugs. Before any expensive lab work, computational screening helps flag which molecules are worth pursuing and which have obvious problems. This app is a small, educational version of that idea.

Give it a molecule and it will:

1. Show the molecule's 2D structure, common name, chemical (IUPAC) name, and SMILES.
2. Check it against several drug-likeness rules (Lipinski, Veber, Ghose) and compute a QED drug-likeness score.
3. Predict blood-brain barrier penetration using a machine learning model.
4. Predict CYP2D6 enzyme inhibition (a metabolism-related property) using a second machine learning model.
5. Flag problematic or reactive substructures (PAINS and Brenk alerts).
6. Estimate how hard the molecule would be to synthesise.

Results are organised under the ADMET headings used in real drug discovery, with each output clearly framed as a screening estimate rather than a verdict.

---

## The ADMET Framework

ADMET is how the pharmaceutical field thinks about whether a molecule can become a usable drug. It stands for:

- **Absorption:** does it get into the body?
- **Distribution:** where does it travel once inside?
- **Metabolism:** how is it broken down?
- **Excretion:** how is it cleared out?
- **Toxicity:** is it harmful?

The app maps its features onto these headings. It covers Absorption, Distribution, Metabolism (partially), and Toxicity (as substructure flags), and is explicit that Excretion is not implemented. This honest mapping, including the gaps, mirrors how a real screening tool's scope would be described.

---

## Features in Detail

### Absorption and Drug-Likeness

Rule-based checks computed directly with RDKit, indicating whether the molecule resembles an orally absorbed drug:

- **Lipinski's Rule of Five:** molecular weight, LogP, hydrogen bond donors and acceptors.
- **Veber's Rules:** rotatable bonds and topological polar surface area (TPSA).
- **Ghose Filter:** molecular weight, LogP, molar refractivity, and atom count.
- **QED (Quantitative Estimate of Drug-likeness):** a single 0-to-1 score summarising overall drug-likeness.

These are guidelines, not pass/fail verdicts. Many real, useful drugs violate one or more of them.

### Distribution

A machine learning model predicts whether the molecule can cross the **blood-brain barrier**, which matters for any drug intended to act on the central nervous system. Output is a likely/unlikely call with a confidence score.

### Metabolism

A machine learning model predicts whether the molecule inhibits **CYP2D6**, one of the cytochrome P450 enzymes responsible for metabolising many drugs. CYP inhibition is a common cause of dangerous drug-drug interactions. This is a screening flag for one enzyme, not a complete metabolism profile.

### Excretion

Not implemented, and the app says so plainly. A clearance/half-life regression model was attempted on a public dataset but had no real predictive power (see Limitations). Rather than display a misleading number, no model is shipped for this property.

### Toxicity and Structural Alerts

Two curated substructure filters (built into RDKit) flag chemical groups associated with problems:

- **PAINS:** substructures known to cause false positives in screening assays.
- **Brenk:** reactive, unstable, or potentially toxic substructures.

These are warnings, not toxicity predictions. A clean result does not guarantee safety, and a flag does not guarantee harm.

### Synthesizability

A **synthetic accessibility (SA) score** estimates how difficult the molecule would be to make in a lab (1 = easy, 10 = very hard). This sits outside ADMET but matters in practice: a molecule that looks perfect but cannot be synthesised is of little use.

---

## The Machine Learning Models

Both models share the same approach: each molecule is converted into a 2048-bit **Morgan fingerprint** (a numeric vector encoding its substructures), and a **Random Forest** classifier is trained on those fingerprints.

| Model | Property | Dataset | Size | Headline metric |
|---|---|---|---|---|
| Blood-brain barrier | Distribution | MoleculeNet BBBP (~2,000 molecules) | binary | Accuracy 0.89, AUC-ROC 0.93 |
| CYP2D6 inhibition | Metabolism | TDC CYP2D6 (Veith) (~13,000 molecules) | binary | Accuracy 0.85, AUC-ROC 0.86 |

Notes on the CYP2D6 model: the dataset is imbalanced (about 19% inhibitors), so the model uses balanced class weights and is evaluated primarily on AUC-ROC rather than accuracy (which would be misleadingly high for a model that simply predicted "non-inhibitor" for everything). The model is intentionally conservative, with few false alarms but catching roughly 63% of true inhibitors, so an "inhibitor" prediction is stronger evidence than a "non-inhibitor" one. Both models were pruned (limited tree depth) to keep file sizes small and reduce overfitting.

All metrics come from held-out test sets with a fixed random seed. They reflect single train/test splits; cross-validation would give more stable estimates.

---

## The Chemistry

**SMILES strings** are a text format for molecular structure (for example, caffeine is `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`). Every molecule enters the app as a SMILES string, whether typed directly, looked up by name via PubChem, or selected from the built-in library.

**Morgan fingerprints** convert a molecule into a fixed-length bit vector by recording the local substructure around each atom out to a set radius (here, radius 2, 2048 bits). This gives every molecule a uniform numeric representation that the machine learning models can use.

**Rule-based descriptors** (molecular weight, LogP, TPSA, hydrogen bond counts, molar refractivity, QED, SA score) and the **PAINS/Brenk substructure catalogs** are all computed directly by RDKit, with no machine learning involved.

---

## Project Structure

```
drug-likeness-predictor-v2/
├── app.py                       # The Streamlit web app
├── random_forest_model.pkl      # Trained blood-brain barrier model
├── cyp2d6_model.pkl             # Trained CYP2D6 inhibition model
├── requirements.txt             # Python dependencies (with pinned versions)
├── packages.txt                 # System libraries needed for RDKit structure drawing
└── README.md                    # This file
```

The models were trained in a Google Colab notebook and saved with joblib. The app loads them at runtime.

A note on `requirements.txt`: scikit-learn, numpy, and joblib are pinned to the exact versions used to train and save the models. This matters because a model saved with one version of scikit-learn will fail to load under a different version. Pinning keeps the deployment environment matched to the training environment.

A note on `packages.txt`: RDKit's molecule drawing depends on system graphics libraries (`libxrender1`, `libxext6`) that are not Python packages and are not present on a bare server by default. This file installs them so structure rendering works in deployment.

---

## How to Run It

### Live version

The app is deployed on Streamlit Community Cloud: **https://predictor-v2.streamlit.app/**

(Free Streamlit apps sleep after a period of inactivity and wake up a few seconds after the link is opened.)

### Run locally

Requires Python 3.11.

```bash
git clone https://github.com/raghavpushkar/drug-likeness-predictor-v2.git
cd drug-likeness-predictor-v2

python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

The app opens at `http://localhost:8501`. (Structure drawing and PubChem name lookup may need the system libraries listed in `packages.txt`, which are installed automatically on Streamlit Cloud but may need manual installation locally on some systems.)

### Using the app

1. Choose an input method: pick from the library, search by name, or enter a SMILES string.
2. Click **Screen molecule**.
3. Review the structure, identity, and the ADMET breakdown.

Molecules worth trying: **fluoxetine** (a known CYP2D6 inhibitor, so the Metabolism flag fires), **caffeine** (passes most filters, crosses into the brain), and **atorvastatin** or **cholesterol** (large or greasy molecules that fail several drug-likeness rules).

---

## Limitations and Honest Caveats

- **This is an educational project, not a real drug-discovery or medical tool.** No output should inform medical, clinical, or research decisions.
- **The ML models are screening estimates.** BBB and CYP2D6 prediction reach useful but imperfect accuracy (AUC around 0.86 to 0.93). They will misclassify some molecules.
- **Excretion was attempted and deliberately not shipped.** A clearance regression model trained on the TDC hepatocyte clearance dataset (~1,200 compounds) achieved R² ≈ 0.06, meaning essentially no predictive power. This reflects a genuine difficulty: clearance is very hard to predict from molecular structure alone on small public datasets. Displaying such a model's output would be misleading, so it was left out.
- **Rule-based filters are guidelines.** Lipinski, Veber, Ghose, PAINS, and Brenk encode useful heuristics, not definitive judgments. Useful drugs routinely violate them.
- **Structural alerts are not toxicity predictions.** PAINS and Brenk flag patterns of concern; a clean molecule is not proven safe, and a flagged molecule is not proven toxic. Validated toxicity prediction would require models trained on data such as Tox21 or ClinTox.
- **Metrics are from single train/test splits.** Cross-validation would give more robust estimates.
- **Name lookup depends on PubChem.** When PubChem is unreachable or lacks an entry, name fields are omitted, though the SMILES and all predictions still work.

---

## Possible Improvements

- **More metabolism coverage:** additional CYP enzymes (CYP3A4, CYP2C9, and others) beyond CYP2D6.
- **A real toxicity model:** trained on Tox21, ClinTox, or hERG cardiotoxicity data, to replace the current substructure-only toxicity flags.
- **Cross-validated metrics:** averaged across folds for more stable performance estimates.
- **Solubility and lipophilicity regression:** additional ADMET-relevant properties from public datasets.
- **Probability calibration:** so the confidence scores more accurately reflect true probabilities.

---

*Built as a self-directed learning project in computational chemistry and machine learning.*
