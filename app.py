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
# Page config + clinical/scientific styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ADMET Screener",
    page_icon="🧬",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Theme-adaptive: uses Streamlit's own theme variables so it works in
       both light and dark mode instead of hardcoding light colours. */
    h1, h2, h3 { letter-spacing: -0.01em; }
    .block-container { padding-top: 2.5rem; max-width: 1100px; }
    /* Section header band with a teal accent rule on the left */
    .section-band {
        border-left: 4px solid #14a3b0;
        padding: 0.35rem 0 0.35rem 1rem;
        margin: 1.6rem 0 0.6rem 0;
        border-bottom: 1px solid rgba(128,128,128,0.25);
    }
    .section-band h3 { margin: 0; font-size: 1.15rem; }
    .section-sub { opacity: 0.7; font-size: 0.85rem; margin: 0.1rem 0 0 1rem; }
    /* Result card: semi-transparent so it adapts to light/dark backgrounds */
    .card {
        background: rgba(128,128,128,0.06);
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .rule-line { font-size: 0.92rem; padding: 0.1rem 0; }
    .summary-tag {
        display: inline-block; font-size: 0.78rem; font-weight: 600;
        padding: 0.1rem 0.6rem; border-radius: 12px; margin-left: 0.5rem;
    }
    .tag-pass { background: rgba(20,163,176,0.18); color: #14a3b0; }
    .tag-warn { background: rgba(200,120,30,0.18); color: #c8781e; }
    /* Identity panel for names + SMILES */
    .id-panel { background: rgba(128,128,128,0.06); border: 1px solid rgba(128,128,128,0.25);
                border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.6rem; font-size: 0.9rem; }
    .id-label { opacity: 0.6; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }
    .id-value { margin-bottom: 0.5rem; word-break: break-word; }
    .id-smiles { font-family: monospace; font-size: 0.82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Setup: model, fingerprint generator, substructure catalogs, SA scorer
# ---------------------------------------------------------------------------

@st.cache_resource
def load_resources():
    model = joblib.load("random_forest_model.pkl")
    cyp_model = joblib.load("cyp2d6_model.pkl")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def make_catalog(filter_type):
        params = FilterCatalogParams()
        params.AddCatalog(filter_type)
        return FilterCatalog.FilterCatalog(params)

    pains = make_catalog(FilterCatalogParams.FilterCatalogs.PAINS)
    brenk = make_catalog(FilterCatalogParams.FilterCatalogs.BRENK)

    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    return model, cyp_model, gen, pains, brenk, sascorer

model, cyp_model, generator, pains_catalog, brenk_catalog, sascorer = load_resources()

# ---------------------------------------------------------------------------
# Curated molecule library (name -> SMILES), baked in for instant, reliable use
# ---------------------------------------------------------------------------

MOLECULE_LIBRARY = {
    # Common over-the-counter / everyday
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Paracetamol (Acetaminophen)": "CC(=O)Nc1ccc(O)cc1",
    "Nicotine": "CN1CCC[C@H]1c1cccnc1",
    # CNS drugs (good demos for the BBB model)
    "Diazepam": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
    "Fluoxetine (Prozac)": "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1",
    "Diphenhydramine": "CN(C)CCOC(c1ccccc1)c1ccccc1",
    "Levodopa": "N[C@@H](Cc1ccc(O)c(O)c1)C(=O)O",
    "Morphine": "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
    # Antibiotics / larger molecules (often fail filters)
    "Penicillin G": "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)Cc1ccccc1)C(=O)O)C",
    "Amoxicillin": "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@H](N)c1ccc(O)cc1)C(=O)O)C",
    # Deliberate filter-failers / interesting cases
    "Atorvastatin (Lipitor)": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
    "Cholesterol": "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",
    "Glucose": "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O",
    # Reference / simple
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
}

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

def fetch_names(smiles, known_common=None):
    """Return (common_name, iupac_name) for a SMILES via PubChem.
    known_common is used as a fallback/primary if we already have a name.
    Never raises; returns what it can, omitting the rest."""
    common = known_common
    iupac = None
    try:
        results = pcp.get_compounds(smiles, "smiles")
        if results:
            c = results[0]
            iupac = c.iupac_name
            if not common:
                # synonyms[0] is usually the most common name
                syns = c.synonyms
                if syns:
                    common = syns[0]
    except Exception:
        pass
    return common, iupac

def mol_to_fingerprint(mol):
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((2048,), dtype=int)
    ConvertToNumpyArray(fp, arr)
    return arr

def lipinski_breakdown(mol):
    mw = Descriptors.MolWt(mol); logp = Descriptors.MolLogP(mol)
    hd = Lipinski.NumHDonors(mol); ha = Lipinski.NumHAcceptors(mol)
    checks = {
        "Molecular weight ≤ 500": mw <= 500,
        "LogP ≤ 5": logp <= 5,
        "H-bond donors ≤ 5": hd <= 5,
        "H-bond acceptors ≤ 10": ha <= 10,
    }
    values = {"Molecular weight": round(mw, 1), "LogP": round(logp, 2),
              "H-bond donors": hd, "H-bond acceptors": ha}
    return values, checks, sum(1 for p in checks.values() if not p)

def veber_check(mol):
    rb = rdMolDescriptors.CalcNumRotatableBonds(mol); tpsa = rdMolDescriptors.CalcTPSA(mol)
    checks = {"Rotatable bonds ≤ 10": rb <= 10, "TPSA ≤ 140": tpsa <= 140}
    values = {"Rotatable bonds": rb, "TPSA": round(tpsa, 1)}
    return values, checks, sum(1 for p in checks.values() if not p)

def ghose_check(mol):
    mw = Descriptors.MolWt(mol); logp = Descriptors.MolLogP(mol)
    mr = Descriptors.MolMR(mol); atoms = mol.GetNumAtoms()
    checks = {
        "160 ≤ MW ≤ 480": 160 <= mw <= 480,
        "-0.4 ≤ LogP ≤ 5.6": -0.4 <= logp <= 5.6,
        "40 ≤ Molar refractivity ≤ 130": 40 <= mr <= 130,
        "20 ≤ atom count ≤ 70": 20 <= atoms <= 70,
    }
    return checks, sum(1 for p in checks.values() if not p)

def qed_score(mol):
    return round(QED.qed(mol), 3)

def substructure_alerts(mol):
    pains = [e.GetDescription() for e in pains_catalog.GetMatches(mol)]
    brenk = [e.GetDescription() for e in brenk_catalog.GetMatches(mol)]
    return pains, brenk

def sa_score(mol):
    return round(sascorer.calculateScore(mol), 2)

def tag(violations):
    if violations == 0:
        return '<span class="summary-tag tag-pass">Pass</span>'
    return f'<span class="summary-tag tag-warn">{violations} violation{"s" if violations > 1 else ""}</span>'

def section(title, subtitle):
    st.markdown(f'<div class="section-band"><h3>{title}</h3></div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="section-sub">{subtitle}</p>', unsafe_allow_html=True)

def rule_block(title, checks, violations):
    lines = "".join(
        f'<div class="rule-line">{"✅" if p else "❌"} {r}</div>' for r, p in checks.items()
    )
    st.markdown(
        f'<div class="card"><b>{title}</b> {tag(violations)}{lines}</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🧬 Molecular ADMET & Drug-Likeness Screener")
st.markdown(
    '<p style="color:#5b6b7b; font-size:0.95rem; margin-top:-0.5rem;">'
    "Screens a molecule across drug-discovery filters, organised by the ADMET framework "
    "(Absorption, Distribution, Metabolism, Excretion, Toxicity)."
    "</p>",
    unsafe_allow_html=True,
)
st.caption(
    "Educational project. Not for real medical, clinical, or research decisions. "
    "Rule-based filters are guidelines, not verdicts; the ML model is a learned estimate, not a measurement."
)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

input_mode = st.radio(
    "Choose how to enter a molecule:",
    ["Pick from library", "Search by name", "Enter SMILES"],
    horizontal=True,
)

smiles = None
known_common_name = None
if input_mode == "Pick from library":
    choice = st.selectbox("Molecule", list(MOLECULE_LIBRARY.keys()))
    smiles = MOLECULE_LIBRARY[choice]
    known_common_name = choice
elif input_mode == "Search by name":
    name = st.text_input("Molecule name", "caffeine")
    pending_name = name
else:
    smiles = st.text_input("SMILES string", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")

go = st.button("Screen molecule", type="primary")

# ---------------------------------------------------------------------------
# Run screening
# ---------------------------------------------------------------------------

if go:
    if input_mode == "Search by name":
        with st.spinner("Looking up molecule in PubChem..."):
            smiles = name_to_smiles(pending_name)
        if smiles is None:
            st.error(f"Could not find a molecule named '{pending_name}'. Try another name, pick from the library, or enter a SMILES string.")
            st.stop()
        known_common_name = pending_name

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        st.error("Could not parse that molecule. Please check your input and try again.")
        st.stop()

    # Layout: structure on left, headline metrics on right
    left, right = st.columns([1, 1.3])

    with left:
        img = Draw.MolToImage(mol, size=(330, 330))
        st.image(img, use_container_width=False)

    with right:
        fp = mol_to_fingerprint(mol).reshape(1, -1)
        prediction = model.predict(fp)[0]
        probability = model.predict_proba(fp)[0, 1]
        qed = qed_score(mol)
        sa = sa_score(mol)

        # Identity panel: names + SMILES
        with st.spinner("Fetching molecule identity..."):
            common_name, iupac_name = fetch_names(smiles, known_common_name)
        id_html = '<div class="id-panel">'
        id_html += '<div class="id-label">Common name</div>'
        id_html += f'<div class="id-value">{common_name if common_name else "—"}</div>'
        id_html += '<div class="id-label">Chemical (IUPAC) name</div>'
        id_html += f'<div class="id-value">{iupac_name if iupac_name else "—"}</div>'
        id_html += '<div class="id-label">SMILES</div>'
        id_html += f'<div class="id-value id-smiles">{smiles}</div>'
        id_html += '</div>'
        st.markdown(id_html, unsafe_allow_html=True)

        st.markdown("##### At a glance")
        m1, m2 = st.columns(2)
        m1.metric("QED drug-likeness", f"{qed}", help="0 = poor, 1 = excellent")
        m2.metric("Synthetic accessibility", f"{sa}", help="1 = easy, 10 = very hard")
        if prediction == 1:
            st.success(f"BBB penetration: likely ({probability:.0%} confidence)")
        else:
            st.warning(f"BBB penetration: unlikely ({1 - probability:.0%} confidence)")
        cyp_glance = cyp_model.predict(fp)[0]
        if cyp_glance == 1:
            st.warning("CYP2D6: predicted inhibitor")
        else:
            st.success("CYP2D6: predicted non-inhibitor")

    # === ABSORPTION ===
    section("Absorption & Drug-Likeness", "Properties typical of orally absorbed drugs.")
    lip_v, lip_c, lip_n = lipinski_breakdown(mol)
    veb_v, veb_c, veb_n = veber_check(mol)
    gho_c, gho_n = ghose_check(mol)
    rule_block("Lipinski's Rule of Five", lip_c, lip_n)
    rule_block("Veber's Rules", veb_c, veb_n)
    rule_block("Ghose Filter", gho_c, gho_n)
    with st.expander("Computed property values"):
        for k, v in {**lip_v, **veb_v}.items():
            st.write(f"**{k}:** {v}")

    # === DISTRIBUTION ===
    section("Distribution", "Where the molecule travels. Here: blood-brain barrier penetration (ML model).")
    if prediction == 1:
        st.success(f"Likely to penetrate the blood-brain barrier (confidence: {probability:.0%})")
    else:
        st.warning(f"Unlikely to penetrate the blood-brain barrier (confidence: {1 - probability:.0%})")
    st.caption("Random Forest trained on the MoleculeNet BBBP dataset (~90% accuracy). An estimate, not a measurement.")

    # === METABOLISM & EXCRETION ===
    section("Metabolism", "How the molecule is broken down. Here: CYP2D6 enzyme inhibition (ML model).")
    cyp_pred = cyp_model.predict(fp)[0]
    cyp_prob = cyp_model.predict_proba(fp)[0, 1]
    if cyp_pred == 1:
        st.warning(f"Predicted CYP2D6 inhibitor (confidence: {cyp_prob:.0%})")
    else:
        st.success(f"Predicted non-inhibitor of CYP2D6 (confidence: {1 - cyp_prob:.0%})")
    st.caption(
        "Random Forest trained on the TDC CYP2D6 (Veith) dataset, AUC ≈ 0.86. "
        "CYP2D6 is one of several drug-metabolising enzymes; this is a screening flag for "
        "one enzyme, not a full metabolism profile. The model leans conservative (catches "
        "~63% of true inhibitors), so a 'non-inhibitor' result is weaker evidence than an "
        "'inhibitor' one."
    )

    section("Excretion", "")
    st.info(
        "Not implemented. A clearance/half-life regression model was attempted on the TDC "
        "hepatocyte clearance dataset (~1,200 compounds) but achieved R² ≈ 0.06, i.e. no real "
        "predictive power. Clearance is very hard to predict from structure alone on small public "
        "data, so no model is shipped here rather than displaying a misleading number. See the "
        "project's limitations for details."
    )

    # === TOXICITY ===
    section("Toxicity & Structural Alerts", "Substructure flags for problematic or reactive groups. Warnings, not toxicity predictions.")
    pains_hits, brenk_hits = substructure_alerts(mol)
    pains_line = "✅ PAINS (assay-interference): none found" if not pains_hits else f"⚠️ PAINS alerts: {', '.join(pains_hits)}"
    brenk_line = "✅ Brenk (reactive/toxic): none found" if not brenk_hits else f"⚠️ Brenk alerts: {', '.join(brenk_hits)}"
    st.markdown(f'<div class="card"><div class="rule-line">{pains_line}</div><div class="rule-line">{brenk_line}</div></div>', unsafe_allow_html=True)
    st.caption(
        "A clean result does not guarantee safety; a flag does not guarantee toxicity. "
        "Validated toxicity prediction would need models trained on data such as Tox21 or ClinTox."
    )

    # === SYNTHESIZABILITY ===
    section("Synthesizability", "How hard the molecule would likely be to make. Outside ADMET, but matters in practice.")
    st.markdown(f'<div class="card"><b>Synthetic Accessibility (SA) score:</b> {sa} <span style="color:#5b6b7b;">(1 = easy, 10 = very difficult)</span></div>', unsafe_allow_html=True)
