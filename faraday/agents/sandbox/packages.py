
BASE_PACKAGES_SANDBOX = [
    "pandas", "numpy"
]

BIO_PACKAGES_SANDBOX = [
    "biopython",
]
CHEM_PACKAGES_SANDBOX = [
    "rdkit", "datamol", "medchem", "scikit-learn", "scipy",
]

DOCUMENT_PACKAGES_SANDBOX = [
    "PyPDF2", "pdfplumber", "openpyxl", "markdown", # document handling packages
]

ALL_PACKAGES_SANDBOX = BASE_PACKAGES_SANDBOX + \
                       BIO_PACKAGES_SANDBOX + \
                       CHEM_PACKAGES_SANDBOX + \
                       DOCUMENT_PACKAGES_SANDBOX


# ------
CORE_BASE = [
    "pandas", "numpy", "scikit-learn", "scipy", "tqdm", "requests", "matplotlib", "seaborn", "httpx"
]

CORE_DOCUMENT = [
    "PyPDF2", "pdfplumber", "openpyxl", "markdown"
]

CORE_CHEM = [
    "rdkit-pypi",     # rdkit binary wheels (pip)
    "datamol", 
    "mordred",        # descriptors
    "cclib",          # QM parsers
    "pubchempy",      # quick structure/data fetch
    "chembl_webresource_client",
    "medchem"
]

CORE_BIO = [
    "biopython", 
    "mygene",         # gene ID/annotation lookups
    "biotite"         # lightweight structural/sequence utils
]

CORE_SANDBOX = CORE_BASE + CORE_DOCUMENT + CORE_CHEM + CORE_BIO


# ------

EXT_CHEM = [
    "openbabel-wheel",        # PyPI wheel variant; otherwise system openbabel
    "py3Dmol",                # visualization
    "meeko",                  # ligand prep for AutoDock/Gnina
    # "gnina",                # usually not via pip; run as CLI in image if needed
    "networkx"
]

EXT_UTILS = [
    "httpx",                  # async-friendly HTTP
    "python-rapidjson"        # faster JSON when streaming agent events
]


# ---
EXTRA_CHEM = [
    "PaDELpy",               # QSAR descriptors
    "rdchiral",              # reaction templates
    "rxnmapper",             # atom mapping
    "molfeat",               # featurization
    "pubchempy",             # PubChem API
    "chembl_webresource_client",
    "fpsim2",                # fast fingerprint search
    "molvs",                 # standardization
    "pdb-tools"              # PDB handling
]


EXTENDED_SANDBOX = CORE_SANDBOX + EXT_CHEM + EXT_UTILS
