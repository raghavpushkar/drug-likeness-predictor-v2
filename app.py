import streamlit as st
import joblib
import numpy as np
import os
import sys
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Draw, QED, rdMolDescriptors, RDConfig
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem import FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams

# ---------------------------------------------------------------------------
# Setup: load model, fingerprint generator, substructure catalogs, SA scorer
# ---------------------------------------------------------------------------

model = joblib.load("random_forest_model.pkl")
generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def make_catalog(filter_type):
    params = FilterCatalogParams()
    params.AddCatalog(filter_type)
    return FilterCatalog.FilterCatalog(params)

pains_catalog = make_catalog(FilterCatalogParams.FilterCatalogs.PAINS)
brenk_catalog = make_catalog(FilterCatalogParams.FilterCatalogs.BRENK)

# Synthetic accessibility scorer (ships inside RDKit's Contrib folder)
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def name_to_smiles(name):
    try:
        results = pcp.get_compounds(name, "name")
        if results:
            return results[0].connectivity_smiles
        return None
    except Exception:
        return None

def mol_to_fingerprint(mol):
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((2048,), dtype=int)
    ConvertToNumpyArray(fp, arr)
    return arr

def lipinski_breakdown(mol):
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    h_donors = Lipinski.NumHDonors(mol)
    h_acceptors = Lipinski.NumHAcceptors(mol)
    checks = {
        "Molecular weight <= 500": mw <= 500,
        "LogP <= 5": logp <= 5,
        "H-bond donors <= 5": h_donors <= 5,
        "H-bond acceptors <= 10": h_acceptors <= 10,
    }
    values = {
        "Molecular weight": round(mw, 1),
        "LogP": round(logp, 2),
        "H-bond donors": h_donors,
        "H-bond acceptors": h_acceptors,
    }
    violations = sum(1 for passed in checks.values() if not passed)
    return values, checks, violations

def veber_check(mol):
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    checks = {
        "Rotatable bonds <= 10": rot_bonds <= 10,
        "TPSA <= 140": tpsa <= 140,
    }
    values = {
        "Rotatable bonds": rot_bonds,
        "TPSA": round(tpsa, 1),
    }
    violations = sum(1 for passed in checks.values() if not passed)
    return values, checks, violations

def ghose_check(mol):
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    mr = Descriptors.MolMR(mol)
    atoms = mol.GetNumAtoms()
    checks = {
        "160 <= MW <= 480": 160 <= mw <= 480,
        "-0.4 <= LogP <= 5.6": -0.4 <= logp <= 5.6,
        "40 <= Molar refractivity <= 130": 40 <= mr <= 130,
        "20 <= atom count <= 70": 20 <= atoms <= 70,
    }
    violations = sum(1 for passed in checks.values() if not passed)
    return checks, violations

def qed_score(mol):
    return round(QED.qed(mol), 3)

def substructure_alerts(mol):
    pains_hits = [entry.GetDescription() for entry in pains_catalog.GetMatches(mol)]
    brenk_hits = [entry.GetDescription() for entry in brenk_catalog.GetMatches(mol)]
    return pains_hits, brenk_hits

def sa_score(mol):
    return round(sascorer.calculateScore(mol), 2)

def rule_summary(violations):
    """Return a short pass/fail label for a rule set based on violation count."""
    if violations == 0:
        return "Pass"
    elif violations == 1:
        return "1 violation"
    else:
        return f"{violations} violations"

# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

st.title("Molecular ADMET & Drug-Likeness Screener")
st.write(
    "Enter a molecule by name (e.g. caffeine) or as a SMILES string. "
    "The tool screens it across several drug-discovery filters, organised by the "
    "ADMET framework (Absorption, Distribution, Metabolism, Excretion, Toxicity)."
)
st.caption(
    "Educational project. Not for real medical, clinical, or research decisions. "
    "Rule-based filters are guidelines, not verdicts, and the ML model is a learned "
    "estimate, not a measurement."
)

input_mode = st.radio("Input type:", ["Molecule name", "SMILES string"])

if input_mode == "Molecule name":
    user_input = st.text_input("Molecule name", "caffeine")
else:
    user_input = st.text_input("SMILES string", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")

if st.button("Screen molecule"):
    # Resolve input into a SMILES string
    if input_mode == "Molecule name":
        with st.spinner("Looking up molecule..."):
            smiles = name_to_smiles(user_input)
        if smiles is None:
            st.error(f"Could not find a molecule named '{user_input}'. Try another name or use a SMILES string.")
            st.stop()
    else:
        smiles = user_input

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Could not parse that molecule. Please check your input and try again.")
        st.stop()

    # --- Structure ---
    st.subheader("Structure")
    img = Draw.MolToImage(mol, size=(350, 350))
    st.image(img, caption=smiles)

    # =======================================================================
    # ABSORPTION & DRUG-LIKENESS
    # =======================================================================
    st.header("Absorption & Drug-Likeness")
    st.caption("Whether the molecule has properties typical of orally absorbed drugs.")

    lip_values, lip_checks, lip_violations = lipinski_breakdown(mol)
    veb_values, veb_checks, veb_violations = veber_check(mol)
    ghose_checks, ghose_violations = ghose_check(mol)
    qed = qed_score(mol)

    st.markdown(f"**Lipinski's Rule of Five** — {rule_summary(lip_violations)}")
    for rule, passed in lip_checks.items():
        st.write(f"{'✅' if passed else '❌'} {rule}")

    st.markdown(f"**Veber's Rules** — {rule_summary(veb_violations)}")
    for rule, passed in veb_checks.items():
        st.write(f"{'✅' if passed else '❌'} {rule}")

    st.markdown(f"**Ghose Filter** — {rule_summary(ghose_violations)}")
    for rule, passed in ghose_checks.items():
        st.write(f"{'✅' if passed else '❌'} {rule}")

    st.markdown(f"**QED (Quantitative Estimate of Drug-likeness):** {qed}  _(0 = poor, 1 = excellent)_")

    with st.expander("See computed property values"):
        all_values = {**lip_values, **veb_values}
        for prop, val in all_values.items():
            st.write(f"**{prop}:** {val}")

    # =======================================================================
    # DISTRIBUTION
    # =======================================================================
    st.header("Distribution")
    st.caption("Where the molecule travels in the body. Here: blood-brain barrier penetration (ML model).")

    fp = mol_to_fingerprint(mol).reshape(1, -1)
    prediction = model.predict(fp)[0]
    probability = model.predict_proba(fp)[0, 1]

    if prediction == 1:
        st.success(f"Likely to penetrate the blood-brain barrier (confidence: {probability:.0%})")
    else:
        st.warning(f"Unlikely to penetrate the blood-brain barrier (confidence: {1 - probability:.0%})")
    st.caption("Random Forest trained on the MoleculeNet BBBP dataset (~90% accuracy). An estimate, not a measurement.")

    # =======================================================================
    # METABOLISM & EXCRETION
    # =======================================================================
    st.header("Metabolism & Excretion")
    st.info(
        "Not yet implemented. A full ADMET tool would predict metabolic stability "
        "(e.g. cytochrome P450 interactions) and clearance/half-life here, which "
        "would require additional trained models. Noted explicitly rather than left "
        "silently blank."
    )

    # =======================================================================
    # TOXICITY & STRUCTURAL ALERTS
    # =======================================================================
    st.header("Toxicity & Structural Alerts")
    st.caption("Substructure-based flags for problematic or reactive groups. These are warnings, not toxicity predictions.")

    pains_hits, brenk_hits = substructure_alerts(mol)

    if not pains_hits:
        st.write("✅ PAINS (assay-interference substructures): none found")
    else:
        st.write(f"⚠️ PAINS alerts: {', '.join(pains_hits)}")

    if not brenk_hits:
        st.write("✅ Brenk (reactive/toxic substructures): none found")
    else:
        st.write(f"⚠️ Brenk alerts: {', '.join(brenk_hits)}")

    st.caption(
        "A clean result does not guarantee safety; a flagged result does not guarantee "
        "toxicity. Validated toxicity prediction would need models trained on data such "
        "as Tox21 or ClinTox."
    )

    # =======================================================================
    # SYNTHESIZABILITY (outside ADMET)
    # =======================================================================
    st.header("Synthesizability")
    st.caption("How hard the molecule would likely be to make in a lab. Sits outside ADMET but matters for real drug development.")

    sa = sa_score(mol)
    st.markdown(f"**Synthetic Accessibility (SA) score:** {sa}  _(1 = easy to synthesise, 10 = very difficult)_")
