#EDA
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
__________________________________________________________________________________________________
#Cleaned
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
