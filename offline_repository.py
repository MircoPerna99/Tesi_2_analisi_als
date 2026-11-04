from pathlib import Path

import pandas as pd


class OfflineRepository:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.interactions = self._read("als_interactions.pkl")
        self.gold_standard = self._read("global_dti.pkl")
        self.drugs = self._read("drug_structures.pkl")
        self.proteins = self._read("protein_structures.pkl")
        for frame, columns, name in (
            (self.interactions, {"drug_id", "protein_id", "amount_interactions"}, "als_interactions"),
            (self.gold_standard, {"drug_id", "protein_id"}, "global_dti"),
            (self.drugs, {"drug_id", "drug_structure"}, "drug_structures"),
            (self.proteins, {"protein_id", "protein_structure"}, "protein_structures"),
        ):
            missing = columns.difference(frame.columns)
            if missing:
                raise ValueError(f"{name}: colonne mancanti {sorted(missing)}")
        self.drugs = self.drugs.dropna(subset=["drug_id"])
        self.proteins = self.proteins.dropna(subset=["protein_id"])
        self.drugs["drug_id"] = self.drugs["drug_id"].astype(int)
        self.proteins["protein_id"] = self.proteins["protein_id"].map(self._protein_id)
        for frame, column in ((self.drugs, "drug_structure"), (self.proteins, "protein_structure")):
            frame[column] = frame[column].map(lambda value: value.encode("utf-8") if isinstance(value, str) else value)

    @staticmethod
    def _protein_id(value):
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    def _read(self, name):
        path = self.data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"File offline mancante: {path}. Esegui export_holdout_data.py sulla macchina con il database.")
        return pd.read_pickle(path)

    def read_interactions_for_als_2(self):
        return self.interactions.copy()

    def read_global_dti(self):
        return self.gold_standard.copy()

    def take_structures(self, ids):
        pairs = pd.DataFrame([(row[0], row[1]) for row in ids], columns=["drug_id", "protein_id"])
        if pairs.empty:
            return pd.DataFrame(columns=["drug_id", "drug_structure", "protein_id", "protein_structure"])
        pairs["drug_id"] = pairs["drug_id"].astype(int)
        pairs["protein_id"] = pairs["protein_id"].map(self._protein_id)
        result = pairs.merge(self.drugs, on="drug_id", how="inner", sort=False)
        result = result.merge(self.proteins, on="protein_id", how="inner", sort=False)
        return result[["drug_id", "drug_structure", "protein_id", "protein_structure"]]
