"""
Created on Fri June 9 2025

@author: Daniel Zeiberg <d.zeiberg@northeastern.edu
"""

import re
import sys

protein_letters_3to1 = {'Ala': 'A',
 'Cys': 'C',
 'Asp': 'D',
 'Glu': 'E',
 'Phe': 'F',
 'Gly': 'G',
 'His': 'H',
 'Ile': 'I',
 'Lys': 'K',
 'Leu': 'L',
 'Met': 'M',
 'Asn': 'N',
 'Pro': 'P',
 'Gln': 'Q',
 'Arg': 'R',
 'Ser': 'S',
 'Thr': 'T',
 'Val': 'V',
 'Trp': 'W',
 'Tyr': 'Y'}
residues_three = set(protein_letters_3to1.keys())

header_parts = ['Variant',
            'Prediction',
            'SD',
            'Comment'] # Variant	Prediction	SD	Comment

def valid_line(line: str) -> bool:
    n_parts = len(line.strip().split(","))
    if n_parts != len(header_parts):
        print(f"Invalid line format: expected {len(header_parts)} parts, got {n_parts}", file=sys.stderr)
        return False
    return True

def split_line(line: str) -> dict:
    parts = line.strip().split(",")
    return dict(zip(header_parts, parts))

def validate_variant(variant: str) -> bool:
    pattern = r'^p\.[A-Z][a-z]{2}[0-9]+([A-Z][a-z]{2}|\=)$'
    if not re.match(pattern, variant):
        print(f"Invalid variant format: '{variant}'", file=sys.stderr)
        return False
    reference_base = variant[2:5]
    if reference_base not in residues_three:
        print(f"Invalid reference base in variant: {variant}", file=sys.stderr)
        return False
    if variant[-1] != '=' and variant[-3:] not in residues_three:
        print(f"Invalid variant change in variant: {variant}", file=sys.stderr)
        return False
    return True

def validate_nonzero(prediction: str, name: str) -> bool:
    try:
        value = float(prediction)
    except ValueError:
        print(f"Invalid {name} value: {prediction}", file=sys.stderr)
        return False
    if value < 0:
        print(f"{name.title()} value cannot be negative: {prediction}", file=sys.stderr)
        return False
    return True

def validate_header(header: str) -> bool:
    header_parts_in_file = list(map(str.strip,header.split(",")))
    for partA, partB in zip(header_parts, header_parts_in_file):
        if partA != partB:
            print(f"Header mismatch: expected '{partA}', found '{partB}'", file=sys.stderr)
            return False
    if len(header_parts_in_file) != len(header_parts):
        print(f"Header length mismatch: expected {len(header_parts)}, found {len(header_parts_in_file)}", file=sys.stderr)
        return False
    return True

def validate_file(file_path: str) -> None:
    with open(file_path, 'r') as file:
        header = file.readline().strip()
        if not validate_header(header):
            print("Header validation failed.", file=sys.stderr)
            return
        success = True
        for line in file:
            if not valid_line(line):
                print(f"\tInvalid line format: {line.strip()}", file=sys.stderr)
                success = False
                continue
            data = split_line(line)
            variant = data[header_parts[0]]
            prediction = data[header_parts[1]]
            sd_prediction = data[header_parts[2]]
            
            if not (
                validate_variant(variant) and
                validate_nonzero(sd_prediction, "SD of prediction")
            ):
                print(f"\tValidation failed for line: {line.strip()}", file=sys.stderr)
                success = False
                continue
    if success:
       print("All validations passed successfully.")

if __name__ == "__main__":
    
    
    path = "/mnt/c/Users/Kunny/Research/Project/BiConVarNet/TSC2/UXFactory_model_1.csv"
    validate_file(path)
