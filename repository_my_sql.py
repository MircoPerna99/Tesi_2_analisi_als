from configuration import Configuration
import mysql.connector # type: ignore
import pandas as pd

class RepositoryMySql():
    def __init__(self):
        self._config = Configuration()
          
    def _startConnetion(self):
        return mysql.connector.connect(
                            host=self._config["sqlHost"],
                            user=self._config["sqlUser"],
                            password=self._config["sqlPassword"],
                            database=self._config["sqlDatabase"]
                            )
    
    
    def read_interactions_for_als(self):
        connection = self._startConnetion()
        cursor = connection.cursor()
        
        cursor.execute("CALL DTI_PPI_Interactions()")

        interactions = cursor.fetchall()
        
        return pd.DataFrame(interactions, columns=['drug_id', 'protein_id', 'amount_interactions'])
    
        
    def read_interactions_for_als_2(self):
        connection = self._startConnetion()
        cursor = connection.cursor()
        
        cursor.execute("CALL DTI_PPI_Interactions_2()")

        interactions = cursor.fetchall()
        
        return pd.DataFrame(interactions, columns=['drug_id', 'protein_id', 'amount_interactions'])
    
    def read_interactions_for_als_3(self):
        connection = self._startConnetion()
        cursor = connection.cursor()
        
        cursor.execute("CALL DTI_PPI_Interactions_3()")

        interactions = cursor.fetchall()
        
        return pd.DataFrame(interactions, columns=['drug_id', 'protein_id', 'amount_interactions'])
    
    def take_structures(self, ids:list):
        connection = self._startConnetion()
        cursor = connection.cursor()
        results = []

        cursor.execute("CREATE TEMPORARY TABLE temp_ids (drug_id INT,protein_id VARCHAR(255))")
        cursor.executemany("INSERT INTO temp_ids (drug_id,protein_id) VALUES (%s,%s)", [(id[0], id[1]) for id in ids])


        query = """
            SELECT tid.drug_id,
                    c.Smiles_Structure as 'drug_structure', 
                    tid.protein_id, 
                    tps.smileStructue as 'protein_structure'
            FROM temp_ids tid
            INNER JOIN Compound c ON c.Id = tid.drug_id
            INNER JOIN Thesis_ProteinStructures tps ON tps.proteinId = tid.protein_id;
        """
        cursor.execute(query)
        results = cursor.fetchall()
        
        
        return pd.DataFrame(results, columns=['drug_id', 'drug_structure','protein_id', 'protein_structure'])
    