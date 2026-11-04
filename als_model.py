from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import explode
from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator
from pyspark.sql.functions import when, col
from pyspark.ml.feature import IndexToString, StringIndexer,StringIndexerModel
from pyspark.ml import Pipeline,PipelineModel
from pyspark.sql import SparkSession
from configuration import Configuration
from pyspark.sql.functions import size

class RecommendedSystem():
    def __init__(self):
        self.spark = SparkSession.builder \
                .appName("Applicazione ALS") \
                  .config("spark.driver.memory", "4g").config("spark.executor.memory", "4g").getOrCreate()
        self._config = Configuration()
        
    def init_data(self, data):
        self._data =  self.spark.createDataFrame(data)
        
    def indexing_name(self):
        self.prontein_indexer = StringIndexer(inputCol = "protein_id", outputCol = "protein_id_index").fit(self._data)
        pipeline = Pipeline(stages = [self.prontein_indexer])
        self.data = pipeline.fit( self._data).transform(self._data)
        self.data.show()
    
    def hyperparameter_tuning(self):
        self.indexing_name()
        (training, test) = self.data.randomSplit([0.8, 0.2], seed=23)
        
        regParams = [0.01, 0.1]
        ranks = [25,30,35]
        alphas = [10.0, 20.0, 40.0, 60.0, 80.0, 100.0]
        
        self.aus_regParam = 0.0
        self.aus_rank = 0
        self.aus_alpha = 0.0
        self.aus_rmse = 0.0
        
        for regParam in regParams:
            for rank in ranks:
                for alpha in alphas:
                    aus_als = ALS(maxIter = 10, regParam = regParam, rank = rank, alpha = alpha, userCol = "drug_id",
                                  itemCol = "protein_id_index", ratingCol = "amount_interactions",coldStartStrategy = "drop")
                    
                    aus_model = aus_als.fit(training)
                    predictions = aus_model.transform(test)
                    evaluator = RegressionEvaluator(metricName = "rmse", labelCol = "amount_interactions", predictionCol = "prediction")
                    rmse = evaluator.evaluate(predictions)
                    
                    if(self.aus_rmse == 0.0 or rmse < self.aus_rmse):
                        self.aus_regParam = regParam
                        self.aus_rank = rank
                        self.aus_alpha = alpha
                        self.aus_rmse = rmse
                        self.model = aus_model

                    print("For regParam: {0}, rank:{1}, alpha:{2}, RMSE:{3}".format(regParam, rank, alpha, rmse))
                     
        print("Chosen parameters: regParam: {0}, rank:{1}, alpha:{2}, RMSE:{3}".format(self.aus_regParam, self.aus_rank, self.aus_alpha, self.aus_rmse))          

        prontein_indexer = StringIndexer(inputCol = "protein_id", outputCol = "protein_id_index").fit(self._data)
        als = ALS(maxIter = 10, regParam = self.aus_regParam, rank = self.aus_rank, alpha = self.aus_alpha, userCol = "drug_id",
                                  itemCol = "protein_id_index", ratingCol = "amount_interactions",coldStartStrategy = "drop")
        pipeline = Pipeline(stages = [prontein_indexer,als])
        pipeline_model = pipeline.fit(self._data)
        trained_indexer = pipeline_model.stages[0]
        trained_indexer.write().overwrite().save("/Users/mircoperna/Desktop/ALS/saved_indexer_2")
        trained_als = pipeline_model.stages[1]
        trained_als.write().overwrite().save("/Users/mircoperna/Desktop/ALS/saved_als_model_2")  
    
    def hyperparameter_tuning_binary(self):
        self.indexing_name()
        (training, test) = self.data.randomSplit([0.8, 0.2], seed=42)

        regParams = [0.001,0.01, 0.1]
        ranks = [25, 30, 35,50]
        maxIters = [10,20,30]
        alphas = [10.0, 20.0, 40.0, 60.0, 80.0, 100.0]

        self.best_auc = 0.0

        for regParam in regParams:
            for rank in ranks:
                for maxIter in maxIters:
                    for alpha in alphas:

                        als = ALS(
                            maxIter=maxIter,
                            regParam=regParam,
                            rank=rank,
                            alpha=alpha,
                            userCol="drug_id",
                            itemCol="protein_id_index",
                            ratingCol="amount_interactions",      # colonna binaria 0/1
                            coldStartStrategy="drop"
                        )

                        model = als.fit(training)
                        predictions = model.transform(test)

                        # Trasforma lo score continuo in predizione binaria
                        predictions = predictions.withColumn(
                            "prediction_binary",
                            when(col("prediction") >= 0.5, 1.0).otherwise(0.0)
                        )

                        evaluator = BinaryClassificationEvaluator(
                            labelCol="amount_interactions",
                            rawPredictionCol="prediction_binary",
                            metricName="areaUnderROC"
                        )

                        auc = evaluator.evaluate(predictions)
                        # Precision
                        precision_evaluator = MulticlassClassificationEvaluator(
                            labelCol="amount_interactions",
                            predictionCol="prediction_binary",
                            metricName="weightedPrecision"
                        )
                        precision = precision_evaluator.evaluate(predictions)

                        # Recall
                        recall_evaluator = MulticlassClassificationEvaluator(
                            labelCol="amount_interactions",
                            predictionCol="prediction_binary",
                            metricName="weightedRecall"
                        )
                        recall = recall_evaluator.evaluate(predictions)

                        # F1-score
                        f1_evaluator = MulticlassClassificationEvaluator(
                            labelCol="amount_interactions",
                            predictionCol="prediction_binary",
                            metricName="f1"
                        )
                        f1 = f1_evaluator.evaluate(predictions)

                        if auc > self.best_auc:
                            self.best_auc = auc
                            self.best_precision = precision
                            self.best_recall = recall
                            self.best_f1 = f1
                            self.aus_regParam = regParam
                            self.aus_rank = rank
                            self.aus_alpha = alpha
                            self.model = model
                            self.best_maxIters = maxIter

                        # print(f"regParam={regParam}, rank={rank}, alpha={alpha}, AUC={auc}")
                        print(f"regParam={regParam}, rank={rank}, alpha={alpha}, AUC={auc}, Precision={precision}, Recall={recall}, F1-score={f1}")


        print(
            f"Chosen parameters: regParam={self.aus_regParam}, "
            f"rank={self.aus_rank}, alpha={self.aus_alpha}, "
            f"AUC={self.best_auc}, Precision={self.best_precision}, Recall={self.best_recall}, F1-score={self.best_f1}, maxIter={self.best_maxIters}"
        )
    
    
    
    def from_index_to_name(self, proteins_recommended):
            prontein_name= IndexToString(inputCol = "protein_id_index", outputCol = "protein_id", labels = self.prontein_indexer.labels)
            pipeline = Pipeline(stages = [prontein_name])
            self.drug_proteins_recommended = pipeline.fit(self.data).transform(proteins_recommended)
            self.drug_proteins_recommended =   self.drug_proteins_recommended.select("drug_id","protein_id","rating")\
                                                                               .orderBy("drug_id","rating")

    def save_pipeline_model_2(self):
        prontein_indexer = StringIndexer(inputCol = "protein_id", outputCol = "protein_id_index").fit(self._data)
        als = ALS(maxIter = 10, regParam = self._config["reg_param"], rank = self._config["rank"], alpha = self._config["alpha"], userCol = "drug_id",
                                  itemCol = "protein_id_index", ratingCol = "amount_interactions",coldStartStrategy = "drop")
        pipeline = Pipeline(stages = [prontein_indexer,als])
        pipeline_model = pipeline.fit(self._data)
        trained_indexer = pipeline_model.stages[0]
        trained_indexer.write().overwrite().save("/Users/mircoperna/Desktop/ALS/saved_indexer_2")
        trained_als = pipeline_model.stages[1]
        trained_als.write().overwrite().save("/Users/mircoperna/Desktop/ALS/saved_als_model_2")                                                                         
    
    def save_pipeline_model(self):
        prontein_indexer = StringIndexer(inputCol = "protein_id", outputCol = "protein_id_index").fit(self._data)
        als = ALS(maxIter = 10, regParam = self._config["reg_param"], rank = self._config["rank"], alpha = self._config["alpha"], userCol = "drug_id",
                                  itemCol = "protein_id_index", ratingCol = "amount_interactions",coldStartStrategy = "drop")
        pipeline = Pipeline(stages = [prontein_indexer,als])
        pipeline_model = pipeline.fit(self._data)
        trained_indexer = pipeline_model.stages[0]
        trained_indexer.write().overwrite().save("/Users/mircoperna/Desktop/ALS/saved_indexer")
        trained_als = pipeline_model.stages[1]
        trained_als.write().overwrite().save("/Users/mircoperna/Desktop/ALS/saved_als_model")
        
    def load_pipeline(self):
        self.als_model = ALSModel.load("/Users/mircoperna/Desktop/ALS/saved_als_model_2")
        self.indexer_model = StringIndexerModel.load("/Users/mircoperna/Desktop/ALS/saved_indexer_2")

    
    def calculate_recommended_proteins(self, number_of_protein_for_drug = 5):
        recommended_proteins= self.als_model.recommendForAllUsers(number_of_protein_for_drug)
        
        recommended_proteins_exploded = recommended_proteins.select(
                                        "drug_id",
                                        explode("recommendations").alias("rec")
                                    ).select(
                                        "drug_id",
                                        "rec.protein_id_index",
                                        "rec.rating"
                                    )


        convertor = IndexToString(
            inputCol="protein_id_index",
            outputCol="protein_id",
            labels=self.indexer_model.labels
        )

        cleaned_proteins = convertor.transform(recommended_proteins_exploded)

        return cleaned_proteins.select("drug_id", "protein_id", "rating")
    
    def calculate_recommended_for_drug(self, drug_ids, number_of_protein_for_drug = 5):
        data = [(d_id,) for d_id in drug_ids]
    
        # Creiamo il DataFrame con la colonna "drug_id"
        df = self.spark.createDataFrame(data, ["drug_id"])
        recommended_proteins= self.als_model.recommendForUserSubset(df, number_of_protein_for_drug)
        
        recommended_proteins_exploded = recommended_proteins.select(
                                        "drug_id",
                                        explode("recommendations").alias("rec")
                                    ).select(
                                        "drug_id",
                                        "rec.protein_id_index",
                                        "rec.rating"
                                    )


        convertor = IndexToString(
            inputCol="protein_id_index",
            outputCol="protein_id",
            labels=self.indexer_model.labels
        )

        cleaned_proteins = convertor.transform(recommended_proteins_exploded)

        return cleaned_proteins.select("drug_id", "protein_id", "rating")
    
    def predict_for_pairs(self, pairs_to_check):
        # 1. Assicurati che i modelli siano stati caricati
        # Se non esistono ancora nell'oggetto, chiamiamo load_pipeline()
        if not hasattr(self, 'als_model') or not hasattr(self, 'indexer_model'):
            self.load_pipeline()

        # 2. Creiamo il DataFrame dalla lista di tuple
        # È fondamentale che i nomi delle colonne corrispondano esattamente
        # a quelli che i modelli si aspettano ("drug_id" e "protein_id")
        df_pairs = self.spark.createDataFrame(pairs_to_check, ["drug_id", "protein_id"])

        # 3. Passiamo il DataFrame nello StringIndexerModel
        # Questo creerà la colonna "protein_id_index" richiesta da ALS
        df_indexed = self.indexer_model.transform(df_pairs)

        # 4. Passiamo il DataFrame indicizzato nel modello ALS
        # Questo genererà una nuova colonna chiamata "prediction"
        df_predictions = self.als_model.transform(df_indexed)

        # (Opzionale) Selezioniamo solo le colonne utili e rimuoviamo l'indice temporaneo
        df_final = df_predictions.select("drug_id", "protein_id", "prediction")

        return df_final