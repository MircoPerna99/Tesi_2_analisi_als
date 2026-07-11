from als_model import RecommendedSystem
from repository_my_sql import RepositoryMySql
from data_manager import DataManager
from predictor import Predictor
from utils import collate
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import torch


TOP_PERCENT_VALUES = [1, 5, 10, 30, 50]
RUN_WGNN_DTA = True  # Imposta a True per eseguire WGNN-DTA sulle coppie ALS candidate o sul gold standard
WGNN_DTA_EVALUATION_MODE = "gold_standard_only"  # Opzioni: "als_candidates", "gold_standard_only"
GOLD_STANDARD_SOURCE = "custom_positive_200"  # Opzioni: "pairs_to_check", "global_dti", "custom_positive_200"
TEST_POSITIVE_INTERACTIONS_FILE = "/Users/mircoperna/Desktop/ALS/test_positive_interactions_ids_ratio_1.csv"
CUSTOM_PAIRS_TO_CHECK_SIZE = 100
CUSTOM_FILE_POSITIVES_SIZE = 100
CUSTOM_RANDOM_STATE = 1234
STRUCTURE_QUERY_CHUNK_SIZE = 1000


def run_wgnn_dta_on_als_candidates():
    return RUN_WGNN_DTA and WGNN_DTA_EVALUATION_MODE == "als_candidates"


def run_wgnn_dta_on_gold_standard_only():
    return RUN_WGNN_DTA and WGNN_DTA_EVALUATION_MODE == "gold_standard_only"

pairs_to_check = [
    (15, "P05026"), (15, "P49721"), (118, "P07919"), (125, "P15954"), (24, "P35625"),
    (146, "Q15375"), (9, "P51451"), (131, "Q9UM73"), (134, "O00206"), (63, "P00156"),
    (52, "Q16665"), (417, "P43405"), (24, "Q53HL2"), (24, "Q9H6I2"), (15, "Q9UBE0"),
    (24, "Q9H7Z6"), (24, "O15151"), (9, "Q9UGP5"), (15, "Q9BZ29"), (24, "P78536"),
    (15, "Q14524"), (24, "Q9NVV9"), (24, "Q9H3D4"), (15, "O15144"), (24, "Q15796"),
    (9, "P42338"), (24, "P15927"), (1841, "P49356"), (9, "P56381"), (45, "P56381"),
    (9, "P13489"), (24, "P62888"), (15, "P19525"), (9, "P09622"), (118, "P14927"),
    (15, "P14927"), (24, "P42892"), (24, "Q99496"), (24, "P35222"), (9, "P35222"),
    (52, "Q9GZT9"), (1523, "P00403"), (1066, "P00403"), (125, "P00403"), (9, "P35221"),
    (15, "Q16877"), (24, "Q15843"), (417, "Q05655"), (1094, "Q92731"), (565, "Q92731"),
    (640, "Q92731"), (417, "P40763"), (15, "Q7L804"), (24, "Q96I99"), (325, "Q16623"),
    (15, "P63165"), (24, "P18146"), (417, "Q14289"), (9, "Q96IV0"), (9, "O00459"),
    (15, "Q9UII2"), (15, "Q13418"), (24, "P32969"), (1063, "P53611"), (1066, "P20674"),
    (15, "Q8TF76"), (24, "Q9BQD3"), (24, "P04637"), (77, "P36888"), (1523, "P14854"),
    (15, "Q9Y5K5"), (24, "Q12888"), (9, "O00443"), (424, "Q8IUR0"), (15, "P43405"),
    (77, "P43405"), (24, "P51451"), (417, "P51451"), (24, "Q02543"), (325, "P60880"),
    (9, "P12694"), (15, "P12694"), (9, "P21953"), (15, "P21953"), (24, "P62913"),
    (24, "Q9UI42"), (15, "P54652"), (24, "Q9H0R8"), (15, "P54619"), (9, "P55072"),
    (118, "P31930"), (1107, "P31930"), (561, "P31930"), (15, "Q9NRR6"), (15, "Q9UBT2"),
    (15, "Q8TAA3"), (15, "P59998"), (15, "Q99436"), (118, "P00156"), (15, "P00156"),
    (561, "P00156"), (24, "P02511"), (24, "Q96CA5"), (24, "P62266"), (24, "Q9H8S9"),
    (24, "Q13625"), (417, "P29353"), (24, "Q02878"), (9, "Q9UPN6"), (24, "P22681"),
    (417, "P22681"), (424, "Q9Y5R8"), (24, "P61218"), (286, "P35916"), (59, "P35916"),
    (1742, "P06401"), (9, "Q8WWV3"), (286, "P35968"), (59, "P35968"), (9, "O14949"),
    (1066, "O14949"), (15, "O14949"), (24, "Q13620"), (24, "P62753"), (24, "O15164"),
    (1066, "P07919"), (1094, "P03372"), (565, "P03372"), (640, "P03372"), (1742, "P03372"),
    (146, "P07333"), (24, "Q13191"), (417, "Q13191"), (134, "Q9Y6Y9"), (24, "P26368"),
    (24, "P62829"), (15, "Q9Y478"), (1841, "P49354"), (24, "P30876"), (1282, "Q5FBB7"),
    (15, "P54760"), (567, "P54760"), (24, "Q13490"), (1107, "Q9UDW1"), (9, "Q9UDW1"),
    (1066, "Q9UDW1"), (15, "Q9UDW1"), (1063, "Q92696"), (15, "O95149"), (15, "Q9UGP5"),
    (15, "Q99418"), (24, "P16333"), (417, "P16333"), (15, "Q04917"), (45, "P25705"),
    (24, "O94972"), (24, "Q9NRG4"), (9, "Q08334"), (1523, "P15954"), (1066, "P15954"),
    (15, "Q12906"), (24, "P27694"), (15, "O15530"), (15, "P49327"), (9, "P08238"),
    (15, "P08238"), (118, "P08574"), (15, "P08574"), (561, "P08574"), (9, "P14616"),
    (15, "P14616"), (1282, "P30153")
]


def require_columns(dataframe, columns, name):
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"{name} non contiene le colonne richieste: {missing}")


def decode_if_bytes(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


def normalize_protein_id(value):
    value = decode_if_bytes(value)
    value = str(value).strip()
    if value.startswith("b'") and value.endswith("'"):
        value = value[2:-1]
    if value.startswith('b"') and value.endswith('"'):
        value = value[2:-1]
    return value


def normalize_pair_columns(dataframe):
    normalized = dataframe.copy()
    normalized["drug_id"] = normalized["drug_id"].astype(int)
    normalized["protein_id"] = normalized["protein_id"].apply(normalize_protein_id)
    return normalized


def take_proteins_unique(dataframe):
    require_columns(dataframe, ["protein_id", "protein_structure"], "interactions_with_structures")
    return dataframe[["protein_id", "protein_structure"]].drop_duplicates().values.tolist()


def take_structures_in_chunks(repo, ids):
    frames = []
    total_chunks = int(np.ceil(len(ids) / STRUCTURE_QUERY_CHUNK_SIZE)) if ids else 0
    for chunk_index, start in enumerate(range(0, len(ids), STRUCTURE_QUERY_CHUNK_SIZE), start=1):
        stop = start + STRUCTURE_QUERY_CHUNK_SIZE
        print(f"Recupero strutture WGNN-DTA chunk {chunk_index}/{total_chunks}")
        frames.append(repo.take_structures(ids[start:stop]))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def read_pairs_to_check_gold_standard():
    return normalize_pair_columns(
        pd.DataFrame(pairs_to_check, columns=["drug_id", "protein_id"])
    ).drop_duplicates()


def pair_set(dataframe):
    normalized = normalize_pair_columns(dataframe)
    return set(zip(normalized["drug_id"], normalized["protein_id"]))


def sample_pairs(dataframe, sample_size, random_state):
    if len(dataframe) <= sample_size:
        return dataframe.copy()
    return dataframe.sample(n=sample_size, random_state=random_state).copy()


def read_custom_positive_200_gold_standard():
    pairs_df = read_pairs_to_check_gold_standard()
    pairs_df["gold_standard_origin"] = "pairs_to_check"

    file_df = pd.read_csv(TEST_POSITIVE_INTERACTIONS_FILE)
    require_columns(
        file_df,
        ["drug_id", "protein_id"],
        "test_positive_interactions_ids_ratio_1.csv",
    )
    file_df = file_df.dropna(subset=["drug_id", "protein_id"]).copy()
    file_df = normalize_pair_columns(file_df)
    if "label" in file_df.columns:
        file_df = file_df[file_df["label"].astype(float).eq(1.0)].copy()
    file_df = file_df[["drug_id", "protein_id"]].drop_duplicates()

    pairs_check_set = pair_set(pairs_df)
    file_df = file_df[
        [
            (drug_id, protein_id) not in pairs_check_set
            for drug_id, protein_id in zip(file_df["drug_id"], file_df["protein_id"])
        ]
    ].copy()
    file_df["gold_standard_origin"] = "test_positive_file"

    pairs_sample = sample_pairs(
        pairs_df,
        CUSTOM_PAIRS_TO_CHECK_SIZE,
        CUSTOM_RANDOM_STATE,
    )
    file_sample = sample_pairs(
        file_df,
        CUSTOM_FILE_POSITIVES_SIZE,
        CUSTOM_RANDOM_STATE,
    )

    selected = pd.concat([pairs_sample, file_sample], ignore_index=True)
    selected = selected.drop_duplicates(subset=["drug_id", "protein_id"]).copy()

    target_total = CUSTOM_PAIRS_TO_CHECK_SIZE + CUSTOM_FILE_POSITIVES_SIZE
    if len(selected) < target_total:
        selected_set = pair_set(selected)
        fill_from_pairs = pairs_df[
            [
                (drug_id, protein_id) not in selected_set
                for drug_id, protein_id in zip(pairs_df["drug_id"], pairs_df["protein_id"])
            ]
        ].copy()
        fill_needed = target_total - len(selected)
        fill_sample = sample_pairs(
            fill_from_pairs,
            fill_needed,
            CUSTOM_RANDOM_STATE + 1,
        )
        fill_sample["gold_standard_origin"] = "pairs_to_check_fill"
        selected = pd.concat([selected, fill_sample], ignore_index=True)
        selected = selected.drop_duplicates(subset=["drug_id", "protein_id"]).copy()

    if len(selected) < target_total:
        print(
            f"Attenzione: gold standard custom con {len(selected)} coppie positive "
            f"invece di {target_total}. Non ci sono abbastanza coppie uniche disponibili."
        )

    return selected.head(target_total).reset_index(drop=True)


def read_global_dti_gold_standard(repo):
    connection = repo._startConnetion()
    cursor = connection.cursor()

    cursor.execute("DROP TEMPORARY TABLE IF EXISTS DTI")
    cursor.execute("DROP TEMPORARY TABLE IF EXISTS Proteins_Structure")
    cursor.execute("DROP TEMPORARY TABLE IF EXISTS DTI_Structures")

    cursor.execute("""
        CREATE TEMPORARY TABLE DTI_Structures AS
        SELECT COMP_STRUCTURE, PROTEIN_STRUCTURE
        FROM Thesis_DB_WGNN_DTA_Human tdwdh
        WHERE tdwdh.`RESULT` = 1
        GROUP BY COMP_STRUCTURE, PROTEIN_STRUCTURE
    """)
    cursor.execute("""
        CREATE TEMPORARY TABLE Proteins_Structure AS
        SELECT tps.proteinId, tps.smileStructue
        FROM Thesis_ProteinStructures tps
        GROUP BY tps.proteinId, tps.smileStructue
    """)
    cursor.execute("""
        CREATE TEMPORARY TABLE DTI AS
        SELECT c.Id AS drug_id, ps.proteinId AS protein_id
        FROM DTI_Structures d
        INNER JOIN Proteins_Structure ps ON d.PROTEIN_STRUCTURE = ps.smileStructue
        INNER JOIN Compound c ON c.Smiles_Structure = d.COMP_STRUCTURE
    """)
    cursor.execute("""
        SELECT drug_id, protein_id
        FROM DTI
        GROUP BY drug_id, protein_id
    """)
    rows = cursor.fetchall()
    cursor.close()
    connection.close()

    return normalize_pair_columns(
        pd.DataFrame(rows, columns=["drug_id", "protein_id"])
    ).drop_duplicates()


def load_gold_standard(repo):
    if GOLD_STANDARD_SOURCE == "pairs_to_check":
        return read_pairs_to_check_gold_standard()
    if GOLD_STANDARD_SOURCE == "global_dti":
        return read_global_dti_gold_standard(repo)
    if GOLD_STANDARD_SOURCE == "custom_positive_200":
        return read_custom_positive_200_gold_standard()
    raise ValueError(
        "GOLD_STANDARD_SOURCE deve essere 'pairs_to_check', 'global_dti' "
        "oppure 'custom_positive_200'."
    )


def build_validated_pairs(als_data, gold_standard_pairs):
    require_columns(als_data, ["drug_id", "protein_id"], "als_data")
    require_columns(gold_standard_pairs, ["drug_id", "protein_id"], "gold_standard_pairs")
    als_data = normalize_pair_columns(als_data)
    gold_standard_pairs = normalize_pair_columns(gold_standard_pairs)
    validated = gold_standard_pairs.drop_duplicates(subset=["drug_id", "protein_id"]).copy()
    validated["gold_standard_source"] = GOLD_STANDARD_SOURCE

    als_drugs = set(als_data["drug_id"])
    als_proteins = set(als_data["protein_id"])

    validated["drug_seen"] = validated["drug_id"].isin(als_drugs)
    validated["protein_seen"] = validated["protein_id"].isin(als_proteins)
    validated["warm_start"] = validated["drug_seen"] & validated["protein_seen"]
    validated["cold_start_category"] = np.select(
        [
            ~validated["drug_seen"] & validated["protein_seen"],
            validated["drug_seen"] & ~validated["protein_seen"],
            ~validated["drug_seen"] & ~validated["protein_seen"],
        ],
        ["cold_start_drug", "cold_start_protein", "cold_start_both"],
        default="warm_start",
    )
    return validated


def run_main_ipynb_flow_all_predictions(repo, model, predictor, data_manager):
    model.load_pipeline()
    all_proteins_count = len(model.indexer_model.labels)
    df = model.calculate_recommended_proteins(all_proteins_count)
    df_check = df.toPandas().rename(columns={"rating": "als_prediction"})
    require_columns(df_check, ["drug_id", "protein_id", "als_prediction"], "df ALS")
    df_check = normalize_pair_columns(df_check)
    df_check = df_check.sort_values(
        ["drug_id", "als_prediction"],
        ascending=[True, False],
    ).copy()
    df_check["als_rank"] = df_check.groupby("drug_id").cumcount() + 1
    df_check.attrs["als_requested_predictions_per_drug"] = all_proteins_count

    if not run_wgnn_dta_on_als_candidates():
        df_check["gat_prediction"] = np.nan
        return df_check

    ids = df_check[["drug_id", "protein_id", "als_prediction"]].values.tolist()
    interactions_with_structures = take_structures_in_chunks(repo, ids)
    require_columns(
        interactions_with_structures,
        ["drug_id", "drug_structure", "protein_id", "protein_structure"],
        "interactions_with_structures",
    )

    proteins_to_analize = take_proteins_unique(interactions_with_structures)
    data_manager.protein_graph_construct(proteins_to_analize)
    dataset = data_manager.prepare_set(interactions_with_structures.values.tolist())
    if len(dataset) == 0:
        raise ValueError("Dataset WGNN-DTA vuoto: impossibile creare il DataLoader.")

    predictor.load_model("GAT")
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=500,
        shuffle=False,
        collate_fn=collate,
    )
    prediction = predictor.make_prediction(test_loader)
    require_columns(prediction, ["drug_structure", "protein_id", "prediction"], "prediction WGNN-DTA")

    interactions_with_structures["drug_structure"] = interactions_with_structures["drug_structure"].apply(decode_if_bytes)
    interactions_with_structures["protein_structure"] = interactions_with_structures["protein_structure"].apply(decode_if_bytes)

    scored = pd.merge(
        df_check,
        interactions_with_structures[["drug_id", "protein_id", "drug_structure", "protein_structure"]],
        on=["drug_id", "protein_id"],
        how="inner",
    )
    scored = pd.merge(
        scored,
        prediction.rename(columns={"prediction": "gat_prediction"}),
        on=["drug_structure", "protein_id"],
        how="left",
    )
    require_columns(scored, ["drug_id", "protein_id", "als_rank", "als_prediction", "gat_prediction"], "scored")
    scored.attrs["als_requested_predictions_per_drug"] = all_proteins_count
    return scored


def run_main_ipynb_flow_for_validated_drugs(repo, model, predictor, data_manager, validated_pairs):
    model.load_pipeline()
    all_proteins_count = len(model.indexer_model.labels)
    drug_ids = sorted(validated_pairs["drug_id"].drop_duplicates().tolist())
    df = model.calculate_recommended_for_drug(drug_ids, all_proteins_count)
    df_check = df.toPandas().rename(columns={"rating": "als_prediction"})
    require_columns(df_check, ["drug_id", "protein_id", "als_prediction"], "df ALS validated drugs")
    df_check = normalize_pair_columns(df_check)
    df_check = df_check.sort_values(
        ["drug_id", "als_prediction"],
        ascending=[True, False],
    ).copy()
    df_check["als_rank"] = df_check.groupby("drug_id").cumcount() + 1
    df_check.attrs["als_requested_predictions_per_drug"] = all_proteins_count
    df_check.attrs["validated_drugs_requested"] = len(drug_ids)

    if not run_wgnn_dta_on_als_candidates():
        df_check["gat_prediction"] = np.nan
        return df_check

    ids = df_check[["drug_id", "protein_id", "als_prediction"]].values.tolist()
    interactions_with_structures = take_structures_in_chunks(repo, ids)
    require_columns(
        interactions_with_structures,
        ["drug_id", "drug_structure", "protein_id", "protein_structure"],
        "interactions_with_structures",
    )

    proteins_to_analize = take_proteins_unique(interactions_with_structures)
    data_manager.protein_graph_construct(proteins_to_analize)
    dataset = data_manager.prepare_set(interactions_with_structures.values.tolist())
    if len(dataset) == 0:
        raise ValueError("Dataset WGNN-DTA vuoto: impossibile creare il DataLoader.")

    predictor.load_model("GAT")
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=500,
        shuffle=False,
        collate_fn=collate,
    )
    prediction = predictor.make_prediction(test_loader)
    require_columns(prediction, ["drug_structure", "protein_id", "prediction"], "prediction WGNN-DTA")

    interactions_with_structures["drug_structure"] = interactions_with_structures["drug_structure"].apply(decode_if_bytes)
    interactions_with_structures["protein_structure"] = interactions_with_structures["protein_structure"].apply(decode_if_bytes)

    scored = pd.merge(
        df_check,
        interactions_with_structures[["drug_id", "protein_id", "drug_structure", "protein_structure"]],
        on=["drug_id", "protein_id"],
        how="inner",
    )
    scored = pd.merge(
        scored,
        prediction.rename(columns={"prediction": "gat_prediction"}),
        on=["drug_structure", "protein_id"],
        how="left",
    )
    require_columns(scored, ["drug_id", "protein_id", "als_rank", "als_prediction", "gat_prediction"], "scored")
    scored.attrs["als_requested_predictions_per_drug"] = all_proteins_count
    scored.attrs["validated_drugs_requested"] = len(drug_ids)
    return scored


def run_wgnn_dta_for_pairs(repo, predictor, data_manager, pairs):
    require_columns(pairs, ["drug_id", "protein_id"], "pairs WGNN-DTA")
    pairs = normalize_pair_columns(pairs)
    pairs = pairs[["drug_id", "protein_id"]].drop_duplicates().copy()

    if pairs.empty:
        pairs["gat_prediction"] = np.nan
        pairs["wgnn_rank"] = pd.Series(dtype=int)
        return pairs

    ids = pairs[["drug_id", "protein_id"]].values.tolist()
    interactions_with_structures = take_structures_in_chunks(repo, ids)
    require_columns(
        interactions_with_structures,
        ["drug_id", "drug_structure", "protein_id", "protein_structure"],
        "interactions_with_structures WGNN-DTA",
    )
    interactions_with_structures = normalize_pair_columns(interactions_with_structures)

    proteins_to_analize = take_proteins_unique(interactions_with_structures)
    data_manager.protein_graph_construct(proteins_to_analize)
    dataset = data_manager.prepare_set(interactions_with_structures.values.tolist())
    if len(dataset) == 0:
        raise ValueError("Dataset WGNN-DTA vuoto: impossibile creare il DataLoader.")

    predictor.load_model("GAT")
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=500,
        shuffle=False,
        collate_fn=collate,
    )
    prediction = predictor.make_prediction(test_loader)
    require_columns(prediction, ["drug_structure", "protein_id", "prediction"], "prediction WGNN-DTA")

    interactions_with_structures["drug_structure"] = interactions_with_structures["drug_structure"].apply(decode_if_bytes)
    interactions_with_structures["protein_structure"] = interactions_with_structures["protein_structure"].apply(decode_if_bytes)
    prediction["drug_structure"] = prediction["drug_structure"].apply(decode_if_bytes)
    prediction["protein_id"] = prediction["protein_id"].apply(normalize_protein_id)

    scored = pd.merge(
        interactions_with_structures[["drug_id", "protein_id", "drug_structure", "protein_structure"]],
        prediction.rename(columns={"prediction": "gat_prediction"}),
        on=["drug_structure", "protein_id"],
        how="left",
    )
    scored = normalize_pair_columns(scored)
    scored = scored.sort_values("gat_prediction", ascending=False).copy()
    scored["wgnn_rank"] = range(1, len(scored) + 1)
    scored.attrs["wgnn_pairs_requested"] = len(pairs)
    scored.attrs["wgnn_pairs_scored"] = len(scored)
    return scored


def mark_validated_candidates(candidates, warm_pairs):
    candidates = normalize_pair_columns(candidates)
    warm_pairs = normalize_pair_columns(warm_pairs)
    warm_set = set(zip(warm_pairs["drug_id"], warm_pairs["protein_id"]))
    marked = candidates.copy()
    marked["is_warm_validated"] = [
        (drug_id, protein_id) in warm_set
        for drug_id, protein_id in zip(marked["drug_id"], marked["protein_id"])
    ]
    marked["y_true"] = marked["is_warm_validated"].astype(int)
    return marked


def score_column_for_method(method):
    score_columns = {
        "als": "als_prediction",
        "wgnn_dta": "gat_prediction",
    }
    if method not in score_columns:
        raise ValueError(f"Metodo non supportato: {method}")
    return score_columns[method]


def precision_recall_global_top_percent(candidates, warm_pairs, methods=None):
    rows = []
    total_als_predictions = len(candidates)
    total_gold_standard_positives = len(warm_pairs)
    positives_in_predictions = int(candidates["y_true"].sum())

    if methods is None:
        methods = ["als", "wgnn_dta"]

    for method in methods:
        score_col = score_column_for_method(method)
        ranked = candidates.dropna(subset=[score_col]).sort_values(score_col, ascending=False).copy()
        if ranked.empty:
            continue

        for percent in TOP_PERCENT_VALUES:
            top_n = int(np.ceil(total_als_predictions * percent / 100.0))
            selected = ranked.head(top_n)
            true_positives = int(selected["y_true"].sum())
            predicted_positives = len(selected)
            false_positives = predicted_positives - true_positives
            false_negatives = total_gold_standard_positives - true_positives

            rows.append({
                "method": method,
                "score_column": score_col,
                "top_percent": percent,
                "total_als_predictions": total_als_predictions,
                "top_n_predictions": top_n,
                "predicted_positives": predicted_positives,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "total_gold_standard_positives": total_gold_standard_positives,
                "gold_standard_positives_in_predictions": positives_in_predictions,
                "precision": true_positives / predicted_positives if predicted_positives else 0.0,
                "recall": true_positives / total_gold_standard_positives if total_gold_standard_positives else 0.0,
            })
    return pd.DataFrame(rows)


def auc_global_ranking(candidates, methods=None):
    rows = []
    if methods is None:
        methods = ["als", "wgnn_dta"]

    for method in methods:
        score_col = score_column_for_method(method)
        scored = candidates.dropna(subset=[score_col]).copy()
        if scored.empty:
            continue

        positives = int(scored["y_true"].sum())
        negatives = int(len(scored) - positives)
        auc = np.nan
        if positives > 0 and negatives > 0:
            auc = roc_auc_score(scored["y_true"], scored[score_col])

        rows.append({
            "method": method,
            "score_column": score_col,
            "total_predictions": len(scored),
            "positives_in_gold_standard": positives,
            "negatives_not_in_gold_standard": negatives,
            "auc": auc,
        })
    return pd.DataFrame(rows)


def build_coverage(validated, candidates):
    warm_pairs = validated[validated["warm_start"]].copy()
    category_counts = validated["cold_start_category"].value_counts().to_dict()
    rows = [
        {"metric": "validated_pairs_total", "value": len(validated)},
        {"metric": "gold_standard_source", "value": GOLD_STANDARD_SOURCE},
        {"metric": "validated_pairs_warm_start", "value": int(validated["warm_start"].sum())},
        {"metric": "validated_pairs_cold_start_drug", "value": category_counts.get("cold_start_drug", 0)},
        {"metric": "validated_pairs_cold_start_protein", "value": category_counts.get("cold_start_protein", 0)},
        {"metric": "validated_pairs_cold_start_both", "value": category_counts.get("cold_start_both", 0)},
        {"metric": "validated_drugs_total", "value": validated["drug_id"].nunique()},
        {"metric": "validated_drugs_warm_start", "value": warm_pairs["drug_id"].nunique()},
        {"metric": "als_requested_predictions_per_drug", "value": candidates.attrs.get("als_requested_predictions_per_drug", np.nan)},
        {"metric": "validated_drugs_requested", "value": candidates.attrs.get("validated_drugs_requested", np.nan)},
        {"metric": "als_predictions_returned_total", "value": len(candidates)},
        {"metric": "gold_standard_positives_in_predictions", "value": int(candidates["y_true"].sum())},
    ]
    if "gold_standard_origin" in validated.columns:
        origin_counts = validated["gold_standard_origin"].value_counts().to_dict()
        rows.extend(
            {"metric": f"gold_standard_origin_{origin}", "value": count}
            for origin, count in origin_counts.items()
        )
    return pd.DataFrame(rows)


repo = RepositoryMySql()
predictor = Predictor()
model = RecommendedSystem()
data_manager = DataManager()

als_data = repo.read_interactions_for_als()
print(als_data.count())
gold_standard_pairs = load_gold_standard(repo)
validated_pairs = build_validated_pairs(als_data, gold_standard_pairs)
warm_validated_pairs = validated_pairs[validated_pairs["warm_start"]].copy()

ranked_candidates = run_main_ipynb_flow_for_validated_drugs(
    repo,
    model,
    predictor,
    data_manager,
    validated_pairs,
)
ranked_candidates = mark_validated_candidates(ranked_candidates, warm_validated_pairs)

percent_metrics_parts = [
    precision_recall_global_top_percent(
        ranked_candidates,
        warm_validated_pairs,
        methods=["als"],
    )
]
auc_metrics_parts = [
    auc_global_ranking(
        ranked_candidates,
        methods=["als"],
    )
]
wgnn_positive_candidates = pd.DataFrame()

if run_wgnn_dta_on_als_candidates():
    percent_metrics_parts.append(
        precision_recall_global_top_percent(
            ranked_candidates,
            warm_validated_pairs,
            methods=["wgnn_dta"],
        )
    )
    auc_metrics_parts.append(
        auc_global_ranking(
            ranked_candidates,
            methods=["wgnn_dta"],
        )
    )
elif run_wgnn_dta_on_gold_standard_only():
    wgnn_gold_standard_pairs = normalize_pair_columns(gold_standard_pairs)
    wgnn_positive_candidates = run_wgnn_dta_for_pairs(
        repo,
        predictor,
        data_manager,
        wgnn_gold_standard_pairs,
    )
    wgnn_positive_candidates = mark_validated_candidates(
        wgnn_positive_candidates,
        wgnn_gold_standard_pairs,
    )
    percent_metrics_parts.append(
        precision_recall_global_top_percent(
            wgnn_positive_candidates,
            wgnn_gold_standard_pairs,
            methods=["wgnn_dta"],
        )
    )
    auc_metrics_parts.append(
        auc_global_ranking(
            wgnn_positive_candidates,
            methods=["wgnn_dta"],
        )
    )

percent_metrics = pd.concat(percent_metrics_parts, ignore_index=True)
auc_metrics = pd.concat(auc_metrics_parts, ignore_index=True)
coverage = build_coverage(validated_pairs, ranked_candidates)
if RUN_WGNN_DTA:
    wgnn_coverage = pd.DataFrame([
        {"metric": "wgnn_dta_evaluation_mode", "value": WGNN_DTA_EVALUATION_MODE},
        {"metric": "wgnn_dta_evaluated_only_gold_standard_pairs", "value": run_wgnn_dta_on_gold_standard_only()},
        {"metric": "wgnn_dta_pairs_requested", "value": wgnn_positive_candidates.attrs.get("wgnn_pairs_requested", np.nan)},
        {"metric": "wgnn_dta_pairs_scored", "value": wgnn_positive_candidates.attrs.get("wgnn_pairs_scored", np.nan)},
    ])
    coverage = pd.concat([coverage, wgnn_coverage], ignore_index=True)

ranked_candidates.to_csv("holdout_2_ranked_candidates.csv", index=False)
wgnn_positive_candidates.to_csv("holdout_2_wgnn_dta_positive_candidates.csv", index=False)
percent_metrics.to_csv("holdout_2_precision_recall_top_percent.csv", index=False)
auc_metrics.to_csv("holdout_2_auc.csv", index=False)
coverage.to_csv("holdout_2_coverage.csv", index=False)
validated_pairs.to_csv("holdout_2_validated_pairs_coverage.csv", index=False)
gold_standard_pairs.to_csv("holdout_2_gold_standard_pairs.csv", index=False)

print("Coverage validation pairs:")
print(coverage)
print("Precision/Recall top percent ranking globale:")
print(percent_metrics)
print("AUC ranking globale:")
print(auc_metrics)
