import streamlit as st
import joblib
import numpy as np
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Draw
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray

# Load the saved model once when the app starts
model = joblib.load("random_forest_model.pkl")
generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def name_to_smiles(name):
    try:
        results = pcp.get_compounds(name, "name")
        if results:
            return results[0].connectivity_smiles
        return None
    except Exception as e:
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

st.title("Molecular Drug-Likeness Predictor")
st.write("Enter a molecule by name (e.g. caffeine) or as a SMILES string to predict blood-brain barrier penetration and check Lipinski's Rule of Five.")

input_mode = st.radio("Input type:", ["Molecule name", "SMILES string"])

if input_mode == "Molecule name":
    user_input = st.text_input("Molecule name", "caffeine")
else:
    user_input = st.text_input("SMILES string", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")

if st.button("Predict"):
    # Resolve the input into a SMILES string
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

    # Draw the structure
    st.subheader("Structure")
    img = Draw.MolToImage(mol, size=(350, 350))
    st.image(img, caption=smiles)

    # Predict
    fp = mol_to_fingerprint(mol).reshape(1, -1)
    prediction = model.predict(fp)[0]
    probability = model.predict_proba(fp)[0, 1]

    st.subheader("Prediction")
    if prediction == 1:
        st.success(f"Likely to penetrate the blood-brain barrier (confidence: {probability:.0%})")
    else:
        st.warning(f"Unlikely to penetrate the blood-brain barrier (confidence: {1 - probability:.0%})")

    # Lipinski
    values, checks, violations = lipinski_breakdown(mol)
    st.subheader("Molecular Properties")
    for prop, val in values.items():
        st.write(f"**{prop}:** {val}")

    st.subheader("Lipinski's Rule of Five")
    for rule, passed in checks.items():
        if passed:
            st.write(f"✅ {rule}")
        else:
            st.write(f"❌ {rule}")
    st.write(f"**Total violations:** {violations}")
