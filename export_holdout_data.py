"""Export the database inputs used by Holdout_2 for offline execution."""
import argparse
import json
from pathlib import Path

import pandas as pd

from configuration import Configuration
from repository_my_sql import RepositoryMySql


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="offline_data")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    repo = RepositoryMySql()
    config = Configuration()

    connection = repo._startConnetion()
    cursor = connection.cursor()
    try:
        cursor.execute("CALL DTI_PPI_Interactions_2()")
        interactions = pd.DataFrame(cursor.fetchall(), columns=["drug_id", "protein_id", "amount_interactions"])
        while cursor.nextset():
            pass
        cursor.execute("""CREATE TEMPORARY TABLE DTI_Structures AS
            SELECT COMP_STRUCTURE, PROTEIN_STRUCTURE FROM Thesis_DB_WGNN_DTA_Human
            WHERE `RESULT` = 1 GROUP BY COMP_STRUCTURE, PROTEIN_STRUCTURE""")
        cursor.execute("""CREATE TEMPORARY TABLE Proteins_Structure AS
            SELECT proteinId, smileStructue FROM Thesis_ProteinStructures
            GROUP BY proteinId, smileStructue""")
        cursor.execute("""CREATE TEMPORARY TABLE DTI AS
            SELECT c.Id AS drug_id, ps.proteinId AS protein_id
            FROM DTI_Structures d
            INNER JOIN Proteins_Structure ps ON d.PROTEIN_STRUCTURE = ps.smileStructue
            INNER JOIN Compound c ON c.Smiles_Structure = d.COMP_STRUCTURE""")
        cursor.execute("SELECT drug_id, protein_id FROM DTI GROUP BY drug_id, protein_id")
        gold = pd.DataFrame(cursor.fetchall(), columns=["drug_id", "protein_id"])
        cursor.execute("SELECT Id, Smiles_Structure FROM Compound")
        drugs = pd.DataFrame(cursor.fetchall(), columns=["drug_id", "drug_structure"])
        cursor.execute("SELECT proteinId, smileStructue FROM Thesis_ProteinStructures")
        proteins = pd.DataFrame(cursor.fetchall(), columns=["protein_id", "protein_structure"])
    finally:
        cursor.close()
        connection.close()

    for name, frame in (("als_interactions.pkl", interactions), ("global_dti.pkl", gold),
                        ("drug_structures.pkl", drugs), ("protein_structures.pkl", proteins)):
        frame.to_pickle(output / name)
        print(f"{name}: {len(frame)} righe")
    with (output / "config.json").open("w", encoding="utf-8") as file:
        json.dump({key: config[key] for key in ("reg_param", "rank", "alpha") if key in config}, file, indent=2)
    print(f"Esportazione completata: {output.resolve()}")


if __name__ == "__main__":
    main()
