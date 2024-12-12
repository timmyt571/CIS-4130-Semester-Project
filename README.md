IMDb Review Ratings Prediction
Goal: I plan to predict the user rating on the movie. I will create a regression model that will utilize various features such as the review text, helpful votes, movie name, and release year to make predictions on the viewers rating of the movie.

Motivation: I wanted to find out whether the movies were good or not based on reviews.

🔎 Data Source: IMDb Review Dataset - ebD
This dataset is from Kaggle and I am using version 2 of the dataset which is about 33GB.

This version contains more than 5.5M reviews/ 1.2M spoilers.

Features:

Review ID: Unique to each review (name)
Movie: Represents name of movie/show
Review Summary: Preview of the response provided by the user
Review Date: The date in which the review was posted
Review Rating: 1-10 integer of how the reviewer rated the movie
Helpful Votes: The number of votes indicating whether the review was marked as helpful by other users.
Review Detail: Actual response provided by reviewer
Star Rating of Movie from Reviewer

📖 Summary
Data Acquisition: Download the IMDb dataset directly into bucket on GCS
Exploratory Data Analysis: Produced descriptive statistics using PYspark
Coding and Modeling: Built a completed machine learning pipeline using random forest regressor
Visualizing Results: Visualized prediction results

I first started by obtaining a dataset which I chose was IMDb reviews dataset from Kaggle (link) and creating a Google Cloud Storage (GCS) bucket called “imdb-reviews”. Then created additional folders inside to organize my storage folders. Next, I set up clusters to create virtual instances and using PySpark to perform exploratory data analysis (EDA) on the dataset. This helped me identify key columns and address any null or missing values. I then created code for a cleaning version of the data to remove incomplete data and unnecessary columns. Moving on to feature engineering and modeling, I normalized the data, performed feature engineering, and  chose to use random forest regression modeling to do training and testing. I allocated 70% of the data for training and 30% for testing. The processed data and trained models were stored in the models folder in my imdb-reviews bucket. Lastly, for data visualization, I used libraries like Matplotlib and Seaborn to create four visualizations that highlighted the dataset and its predictions.

Concluding my project and based on the visualizations and the data cleaning. We can predict that the distribution of actual ratings shows a clear peak at 10, indicating a user bias toward higher ratings, while this may not be a number, we can assume that ratings were mostly positive towards the movies. The predicted ratings capture this trend but with a smoother, more spread-out distribution. Although the model performs well in approximating the general pattern, it slightly underestimates the sharpness of the peak for the highest score. This could be due to the model averaging effects or the exclusion of other potentially influential factors such as parameters. To streamline the analysis, I focused on key attributes, removing excess data and working with a representative sample.


🛠️ Notebooks
Language: PySpark, Python
Libraries: io, pandas, numpy, pyspark.sql.types, Pipeline, StringIndexer, OneHotEncoder, VectorAssembler, RandomForestRegressor, RegressionEvaluator, chain, matplotlib, seaborn
