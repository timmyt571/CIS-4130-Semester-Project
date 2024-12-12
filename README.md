# CIS-4130-Semester-Project

APPENDIX A

Getting started/Created Kaggle Token, uploaded it and searched kaggle dataset:
mkdir .kaggle
ls -la
mv kaggle.json .kaggle/
chmod 600 .kaggle/kaggle.json

Installing necessary things:
sudo apt -y install zip
sudo apt -y install python3-pip python3.11-venv
python3 -m venv pythondev
cd pythondev
source bin/activate
pip3 install kaggle
kaggle datasets list

Using CommandLine to download data set from kaggle:
kaggle datasets download -d ebiswas/imdb-review-dataset

The unzip the imdb-review-dataset.zip file using the command:
unzip imdb-review-dataset.zip

Create a bucket named ‘my-bucket’ in the us-central1 region within the project-id-12345 project ID.
gcloud storage buckets create gs://imdbreviews-bucket --project=thisisaproject-434114 --default-storage-class=STANDARD --location=us-central1 --uniform-bucket-level-access

Once the bucket is created, copy a file from the local file system to the new bucket:
gcloud storage cp part-01.json gs://imdbreviews-bucket/landing/part-01.json
gcloud storage cp part-02json gs://imdbreviews-bucket/landing/part-02.json
gcloud storage cp part-03.json gs://imdbreviews-bucket/landing/part-03.json
gcloud storage cp part-04.json gs://imdbreviews-bucket/landing/part-04.json
gcloud storage cp part-05.json gs://imdbreviews-bucket/landing/part-05.json
gcloud storage cp part-06.json gs://imdbreviews-bucket/landing/part-06.json


APPENDIX B
#Using Python to run to load the data set from GCS and produce descriptive statistics about the data.

from google.cloud import storage
from io import StringIO
import pandas as pd

bucket_name = "imdbreviews-bucket"
storage_client = storage.Client()
folder_pattern = "landing/"
blobs = storage_client.list_blobs(bucket_name, prefix=folder_pattern)
filtered_blobs = [blob for blob in blobs if blob.name.endswith('.json')]

#EDA on DataFrame
def perform_eda(df):
    if df.empty:
        print("No data")
        return
    #Number of observations
    num_observations = df.shape[0]
    print(f"Number of observations: {num_observations}")
    #List of variables
    print("List of variables (columns):")
    print(df.columns.tolist())

  missing_fields = df.isnull().sum()
  print("Number of missing fields in each column:")
  print(missing_fields[missing_fields > 0])

  #Statistics
  numeric_stats = df.describe()
  print("\nStatistics for numeric variables:")
  min_values = numeric_stats.loc['min']
  max_values = numeric_stats.loc['max']
  mean_values = numeric_stats.loc['mean']
  std_values = numeric_stats.loc['std']
  print("\nMin values:")
  print(min_values)
  print("\nMax values:")
  print(max_values)
  print("\nMean values:")
  print(mean_values)
  print("\nStandard deviation:")
  print(std_values)
  
  #Text statistics
  text_cols = df.select_dtypes(include=['object'])
  if not text_cols.empty:
      print()
      print("Text data statistics:")
      for col in text_cols.columns:
          if df[col].apply(lambda x: isinstance(x, str)).all():
              df['word_count'] = df[col].apply(lambda x: len(str(x).split()))
              print(f"{col}:")
              print(f" - Number of documents: {df[col].count()}")
              print(f" - Average word count: {df['word_count'].mean()}")
              print(f" - Min word count: {df['word_count'].min()}")
              print(f" - Max word count: {df['word_count'].max()}")
              df.drop('word_count', axis=1, inplace=True)

#Looping through the datafiles
for blob in filtered_blobs:
    print(f"Processing file: {blob.name} with size {blob.size} bytes")
    df = pd.read_json(StringIO(blob.download_as_text()))
    df.info()
    perform_eda(df)
    print()
    print()

#Creating the Histogram

from google.cloud import storage
from io import StringIO
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

source_bucket_name = "imdbreviews-bucket"
storage_client = storage.Client()
folder_pattern = "landing/"
blobs = storage_client.list_blobs(source_bucket_name, prefix=folder_pattern)
filtered_blobs = [blob for blob in blobs if blob.name.endswith('.json')]

#store data from all files in dataframe
all_data = pd.DataFrame()

#process each blob and append to main dataframe
for blob in filtered_blobs:
    print(f"Processing file: {blob.name} with size {blob.size} bytes")
    df = pd.read_json(StringIO(blob.download_as_text()))
    all_data = pd.concat([all_data, df], ignore_index=True)

#calculate word count for each 'review_detail'
all_data['word_count'] = all_data['review_detail'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)
# group by 'rating' and calculate the average word count
rating_word_count = all_data.groupby('rating')['word_count'].mean().reset_index()

#histogram of ratings with the average word count
plt.figure(figsize=(10, 6))
sns.barplot(data=rating_word_count, x='rating', y='word_count', palette='dark')
plt.xlabel('User Rating')
plt.ylabel('Average Word Count in Review Detail')
plt.title('Average Word Count in Review Detail by User Rating (All Files Combined)')
plt.show()



APPENDIX C
#Cleaning and moving files to /cleaned

from google.cloud import storage
from io import StringIO
import pandas as pd

#Source for the files
bucket_name = "imdbreviews-bucket"

#Create a client variable for GCS
storage_client = storage.Client()

#Get a list of the 'blobs' (objects or files) in the bucket
blobs = storage_client.list_blobs(bucket_name, prefix="landing")

#Data cleaning function
def clean_data(df):
    # Fill nulls or remove records with nulls
    df = df.fillna(value={"column_name": "default_value"})
    df = df.dropna()

    return df

#A for loop to go through all of the blobs and process each JSON file
for blob in blobs:
    if blob.name.endswith('.json'):
        print(f"Processing file: {blob.name}")

#CSV content into a DataFrame
        json_data = blob.download_as_text()
        df = pd.read_json(StringIO(json_data))

        #Print DataFrame info 
        df.info()

        #Clean the data by calling the clean_data function
        df = clean_data(df)

        #Writing the cleaned DataFrame to the cleaned folder as a Parquet file
        cleaned_file_path = f"gs://{bucket_name}/cleaned/{blob.name.split('/')[-1].replace('.json', '.parquet')}"
        df.to_parquet(cleaned_file_path, index=False)
        print(f"Cleaned data written to: {cleaned_file_path}")



APPENDIX D

#Creating Features on columns (FEATURE ENGINEERING)

 #%pip install textblob
from google.cloud import storage
from pyspark.ml.feature import Tokenizer, RegexTokenizer, HashingTF, IDF, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql.functions import col, monotonically_increasing_id
from pyspark.ml import Pipeline

df.printSchema()
# Drop columns we will not use at all
df = df.drop("review_id")
df = df.drop("reviewer")
df = df.drop("helpful")

from pyspark.sql.functions import count
# df.groupby("movie").count().show()
# Count frequency and sort
movie_frequency_df = df.groupBy("movie").agg(count("movie").alias("frequency")).orderBy("frequency", ascending=False)
# Show result
movie_frequency_df.show()
print(movie_frequency_df.count())

# Get the top 1000 movies
top_1000_movies_df = movie_frequency_df.limit(1000)

# Filter original DataFrame by doing an inner join with the top_1000_movies_df on 'Movie' column
df = df.join(top_1000_movies_df, "movie")

# Drop the frequency column
df = df.drop("frequency")

# Show the result
df.show()


# Convert Spoler Tag to a double
df = df.withColumn("spoiler_tag", df.spoiler_tag.cast("double"))

indexer_movie = StringIndexer(inputCol="movie", outputCol="movie_index", handleInvalid="keep")
encoder_movie = OneHotEncoder(inputCol="movie_index", outputCol="movie_vector", handleInvalid="keep")

# sentiment analysis function
def sentiment_analysis(some_text):
    sentiment = TextBlob(some_text).sentiment.polarity
    return sentiment

#registering the UDF for sentiment analysis
sentiment_analysis_udf = udf(sentiment_analysis, DoubleType())

#apply the UDF to calculate sentiment for 'review_summary'
df = df.withColumn("review_summary_sentiment", sentiment_analysis_udf(df["review_summary"]))
df = df.withColumn("review_detail_sentiment", sentiment_analysis_udf(df["review_detail"]))

#final feature vector
assembler = VectorAssembler(
    inputCols=[
#        "reviewer_vector",    
        "spoiler_tag",
        "movie_vector",                  
        "review_summary_sentiment",
        "review_detail_sentiment"
    ],
    outputCol="features"
)

#pipeline with updated stages
pipeline = Pipeline(stages=[
    indexer_movie,
    encoder_movie,
    assembler
])

#fit and transform the pipeline on the data
df_transformed = pipeline.fit(df).transform(df)

# Drop unecessary columns
df_transformed = df_transformed.drop("helpful")
df_transformed = df_transformed.drop("review_summary")
df_transformed = df_transformed.drop("review_detail")

df_transformed.select("review_summary_sentiment", "review_detail_sentiment",  "movie_vector", "features").show(10, truncate=False)

df_transformed.cache()

# Save the transformed dataframe in a "features" folder
df_transformed.write.mode("overwrite").parquet(f"gs://imdbreviews-bucket/features/transformed_data_with_features.parquet")


#Creating RandomForest Model

from google.cloud import storage
from pyspark.sql.functions import col
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

sdf = spark.read.parquet("gs://imdbreviews-bucket/features/transformed_data_with_features.parquet")
sdf.show(10, truncate=False)


#RandomForestRegressor
rf = RandomForestRegressor(featuresCol="features", labelCol="rating")

#split data into training and test sets
train_data, test_data = sdf.randomSplit([0.7, 0.3], seed=42)

#set up cross-validation with hyperparameter tuning
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20, 30]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .build()


#evaluate the model
evaluator_rmse = RegressionEvaluator(labelCol="rating", predictionCol="prediction", metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol="rating", predictionCol="prediction", metricName="mae")
evaluator_r2 = RegressionEvaluator(labelCol="rating", predictionCol="prediction", metricName="r2")

cv = CrossValidator(estimator=rf,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator_rmse,  # Evaluator for RMSE
                    numFolds=3)

#train the model
rf_model = cv.fit(train_data)

#make predictions on the test data
predictions = rf_model.transform(test_data)


rmse = evaluator_rmse.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")

#sample predictions
predictions.select("movie","rating", "prediction").show(10, truncate=False)

best_model = rf_model.bestModel  # Best model after cross-validation

#Extract feature importances
feature_importances = best_model.featureImportances

#Print the feature importances
print("Feature Importances: ")
for feature, importance in zip(sdf, feature_importances):
    print(f"{feature}: {importance}")

#Save the trained model to a location 
best_model.save("gs://imdbreviews-bucket/models/imdb_model")


#Save the predictions to models folder
predictions.select("movie","rating", "prediction").write.parquet("gs://imdbreviews-bucket/models/rating_predictions")

APPENDIX E

from pyspark.ml.regression import RandomForestRegressionModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

model_path = "gs://imdbreviews-bucket/models/imdb_model"
rf_model = RandomForestRegressionModel.load(model_path)

# load the test predictions
rating_predictions_path = "gs://imdbreviews-bucket/models/rating_predictions/*"  
predictions = spark.read.parquet(rating_predictions_path)

# convert predictions to Pandas
predictions_df = predictions.select("rating", "prediction").toPandas()

# Visualization 1: Scatter Plot (Prediction vs Actual)
plt.figure(figsize=(10, 6))
sns.scatterplot(x="rating", y="prediction", data=predictions_df, alpha=0.7)
plt.plot([predictions_df["rating"].min(), predictions_df["rating"].max()], 
         [predictions_df["rating"].min(), predictions_df["rating"].max()], 
         color='red', linestyle="--")
plt.title("Prediction vs Actual Rating")
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.grid(True)
plt.show()

# Visualization 2: Residual Distribution
predictions_df["residual"] = predictions_df["rating"] - predictions_df["prediction"]
plt.figure(figsize=(10, 6))
sns.histplot(predictions_df["residual"], kde=True, bins=30, color="purple")
plt.title("Residuals Distribution")
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Visualization 3: Actual Ratings Distribution
plt.figure(figsize=(10, 6))
sns.histplot(predictions_df["rating"], bins=20, kde=True, color="skyblue")
plt.title("Distribution of Actual Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Visualization 4: Sort features by importance
top_features = importances.sort_values(by="importance", ascending=False).head(10)  
# show top 10 features

# top features
plt.figure(figsize=(12, 8))
sns.barplot(x="importance", y="feature", data=top_features, palette="viridis")
plt.title("Top 10 Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Visualization 5: Plot side-by-side histograms for actual and predicted ratings
plt.figure(figsize=(14, 7))

# Plot Actual Ratings Distribution
plt.subplot(1, 2, 1)
sns.histplot(predictions_df['rating'], kde=True, color='skyblue', bins=20, edgecolor='black')
plt.title('Distribution of Actual Ratings')
plt.xlabel('Actual Rating')
plt.ylabel('Frequency')

# Plot Predicted Ratings Distribution
plt.subplot(1, 2, 2)
sns.histplot(predictions_df['prediction'], kde=True, color='orange', bins=20, edgecolor='black')
plt.title('Distribution of Predicted Ratings')
plt.xlabel('Predicted Rating')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
