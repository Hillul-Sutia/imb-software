import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit

# Load dataset
df = pd.read_csv(r"data\raw\BBBP.csv")

# Assume your SMILES column is named 'SMILES'
def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)

    # Handle acyclic molecules separately
    if scaffold.GetNumAtoms() == 0:
        return "ACYCLIC_" + Chem.MolToSmiles(mol)

    return Chem.MolToSmiles(scaffold)

# Generate scaffold identifiers
df["scaffold"] = df["smiles"].apply(get_scaffold)

# Remove invalid SMILES
df = df.dropna(subset=["scaffold"]).reset_index(drop=True)

# Scaffold groups
print('Total Scaffold groups :', df['scaffold'].value_counts())

# Group molecules by scaffold
groups = df["scaffold"].values

# First: 80% development set, 20% test set
split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

dev_idx, test_idx = next(
    split1.split(df, groups = groups)
)

development = df.iloc[dev_idx].copy()
test = df.iloc[test_idx].copy()

# Second: 90% training, 10% validation
# This gives approximately 72:8:20 overall.
split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.10,
    random_state=42
)

train_rel, val_rel = next(
    split2.split(
        development,
        groups=development["scaffold"].values
    )
)

train = development.iloc[train_rel].copy()
validation = development.iloc[val_rel].copy()

# Verify no scaffold overlap
train_scaffolds = set(train["scaffold"])
val_scaffolds = set(validation["scaffold"])
test_scaffolds = set(test["scaffold"])

assert train_scaffolds.isdisjoint(val_scaffolds)
assert train_scaffolds.isdisjoint(test_scaffolds)
assert val_scaffolds.isdisjoint(test_scaffolds)

print("Training:", len(train))
print("Validation:", len(validation))
print("Testing:", len(test))

print("Training percentage:", len(train)/len(df)*100)
print("Validation percentage:", len(validation)/len(df)*100)
print("Testing percentage:", len(test)/len(df)*100)
