# BigData-ProjectTwo
MapReduce implementation using Spark to measure term-term relevance between a query term and all other terms

In order to run application: spark-submit termTermSimilarity.py <textFile> <queryTerm>

File descriptions:
norms.txt is a text file of the norms of all terms
output.csv is a csv file of the tfidf matrix
queryTermSimOutput.txt is an output file of all similarities between queryTerm and all relevant terms
termTermSimilarity.py is the script to run the algorithim
Project2Data.txt is the text file to analyze
tfidfs.txt is a text of all tfidfs per term per docid


