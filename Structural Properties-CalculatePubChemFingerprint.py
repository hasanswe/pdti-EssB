from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
from rdkit import DataStructs
from estate import CalculateEstateFingerprint as EstateFingerprint
import pybel
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.Pharm2D import Generate
from ghosecrippen import GhoseCrippenFingerprint
from PubChemFingerprints import calcPubChemFingerAll

Version=1.0
similaritymeasure=[i[0] for i in DataStructs.similarityFunctions]
#

def CalculateGhoseCrippenFingerprint(mol, count = False):
    """
    Calculate GhoseCrippen Fingerprints
    """
    res = GhoseCrippenFingerprint(mol, count=count)
    return res

def CalculatePubChemFingerprint(mol):
    """
    Calculate PubChem Fingerprints
    """
    res = calcPubChemFingerAll(mol)
    return res

_FingerprintFuncs={'PubChem': CalculatePubChemFingerprint}
                 
#
if __name__=="__main__":
    
    print '-'*10+'START'+'-'*10
    
    ms = [Chem.MolFromSmiles('CCOC=N'), Chem.MolFromSmiles('NC1=NC(=CC=N1)N1C=CC2=C1C=C(O)C=C2')]
    m2 = [pybel.readstring("smi",'CCOC=N'),pybel.readstring("smi",'CCO')]
    mol = Chem.MolFromSmiles('O=C1NC(=O)NC(=O)C1(C(C)C)CC=C')
    res = Calculate(mol)
    print res7
    print '-'*10+'END'+'-'*10


//Calculating fingerprint via functions

The CalculatePubChemFingerprint() function calculates the PubChem Fingerprint.

>>> from PyBioMed.PyMolecule.fingerprint import CalculatePubChemFingerprint
>>> import pybel
>>> smi = 'CCC1(c2ccccc2)C(=O)N(C)C(=N1)O'
>>> mol = pybel.readstring("smi", smi)
>>> mol_fingerprint = CalculateCalculatePubChemFingerprint(mol)
>>> print len(mol_fingerprint[1])