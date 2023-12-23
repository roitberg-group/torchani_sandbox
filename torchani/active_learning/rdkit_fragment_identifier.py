# EVENTUALLY MERGE THIS INTO STRUCTURE_SANITIZER.PY
from rdkit import Chem
from rdkit.Chem import AllChem

mol_path = 'C2H3N7O2_output.mol'

def process_molecule(file_path):
    mol = Chem.MolFromMolFile(file_path)
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    for i, fragment in enumerate(fragments):
        frag_with_h = Chem.AddHs(fragment)
        num_conformers = 5
        AllChem.EmbedMultipleConfs(frag_with_h, numConfs=num_conformers)

        for conf in frag_with_h.GetConformers():
            AllChem.MMFFOptimizeMolecule(frag_with_h, conId=conf.GetId())
        for j, conf in enumerate(frag_with_h.GetConformers()):
            xyz_filename = f"{file_path.stem}_frag_{i}_conf_{j}.xyz"
            Chem.MolToXYZFile(frag_with_h, xyz_filename, confId=conf.GetId())
        print(f"Processed Fragment {i+1} of {file_path}")

process_molecule(mol_path)

