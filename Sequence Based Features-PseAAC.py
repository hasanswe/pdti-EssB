import re

AALetter = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]


#
def CalculateAAComposition(ProteinSequence):
    """
#
    Calculate the composition of Amino acids
    for a given protein sequence.
    Usage:
    result=CalculatePseAAComposition(protein)
    Input: protein is a pure protein sequence.
    Output: result is a dict form containing the composition of
    20 amino acids.
#
    """
    LengthSequence = len(ProteinSequence)
    Result = {}
    for i in AALetter:
        Result[i] = round(float(ProteinSequence.count(i)) / LengthSequence * 100, 3)
    return Result



#
def CalculatePseAAComposition(ProteinSequence):
    """
#
    Calculate the composition of AAC for
    given protein sequence.
    Usage:
    result=CalculateAAComposition(protein)
    Input: protein is a pure protein sequence.
    Output: result is a dict form containing all composition values of
    AAC
#
    """

    result = {}
    result.update(CalculatePseAAComposition(ProteinSequence))
    return result


#
if __name__ == "__main__":
    protein = "ADGCGVGEGTGQGPMCNCMCMKWVYADEDAADLESDSFADEDASLESDSFPWSNQRVFCSFADEDAS"

    AAC = Calculate PseAAComposition(protein)
    print PseAAComposition
    res = CalculatePseAAComposition(protein)
    print len(res)


#Calculate protein descriptors via object


>>> from PyBioMed import Pyprotein
>>> protein="ADGCGVGEGTGQGPMCNCMCMKWVYADEDAADLESDSFADEDASLESDSFPWSNQRVFCSFADEDAS"
>>> protein_class = Pyprotein.PyProtein(protein)
>>> print len(protein_class.GetDPComp())
400