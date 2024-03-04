# Movie-Plot-IR-Mini-Project

                      ...MOVIE PLOT INFORMATIONAL RETRIEVAL PROJECT...

This is Movie Plot Information Retrieval Project which is based on the searching of the movie plot in a given dataset.
The goal of our project is to analyze and extract meaningful information from IMDB dataset.csv dataset.
Our information retrieval algorithm returns similar movie titles based on an input query.

## Project Description

This project is an Information Retrieval system built using Flask, Python, and various libraries for natural language processing and machine learning. The system enables users to search for movie plots based on their queries and provides a feedback mechanism to enhance result relevance.

### Components

Flask Web Application:
The project features a user-friendly Flask web application where users can input search queries.
The system retrieves relevant movie plots from the dataset based on the user's input.

### Data Preprocessing:
Movie dataset preprocessing involves cleaning and stemming the text to enhance data quality.
Preprocessed data is used to create a TF-IDF matrix crucial for calculating document similarities.

### TF-IDF Vectorization:
The project employs the TF-IDF vectorization technique to represent word importance in documents.
This vectorization is essential for calculating the similarity between user queries and movie plots.

### Similarity Calculation:
Cosine similarity measures the similarity between user queries and movie plots using TF-IDF representation.
This determines the relevance of each movie plot to the given query.

### Feedback Mechanism:
The system includes a feedback mechanism allowing users to provide feedback on the relevance of retrieved results.
This feedback is utilized to improve the precision and recall of the information retrieval system.
How to Run the System

### Install Python and Necessary Libraries:

Use the following command to install the required libraries from the requirements.txt file:
pip install -r requirements.txt

After this 

### Run the Main Application:

And for dataset Link : "https://drive.google.com/file/d/1CKwVcjFXs2Jvbipspa6AWQZcymRRdX6F/view?usp=sharing";

### !important:
If "tfidf_matrix.npz," "vocabulary.txt," and "processed_text.csv" files are absent, the project will process the dataset again and server takes some time to start.
If those files are already present then server won't take much time to start
so for those files Link : "https://drive.google.com/drive/folders/1ZOTIWhdFp4Wa0UnD64SSlIn4iOB87UE7?usp=sharing"

Note : project will run without these file also :"tfidf_matrix.npz," "vocabulary.txt," and "processed_text.csv"

Execute the main.py file to run the application.command "python main.py"


### Access the System:

After running the main.py it will give localhost url
                     (or)
Open a web browser and navigate to http://127.0.0.1:5000/ to use the Information Retrieval system.

## Functionalities

### Search Functionality:
Enter a query in the provided input field.
Submit the query to retrieve relevant movie plots.
Example Query:
"A bartender is working at a saloon, serving drinks to customers. After he fills a stereotypically Irish man's bucket with beer, Carrie Nation and her followers burst inside..."

### Feedback Functionality:
Mark movie plots as relevant or not relevant using the provided feedback buttons (Tick Symbol).
Click the Tick Symbol if the document is relevant.
Click "Submit Feedback" to send feedback to the system.
View recall and precision metrics based on provided feedback.
Relevant documents move to the top of the table, and non-relevant ones go to the bottom based on feedback.

## Additional Information

The system automatically processes and saves text data if not already preprocessed.
TF-IDF matrices and vocabulary are saved for efficient retrieval in subsequent runs.
The project includes robust error handling for cases such as missing files or incorrect data structures.