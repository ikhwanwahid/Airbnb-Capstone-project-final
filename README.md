# Capstone Project : Classify suspicious Airbnb Listings (NY) Improving overall UX !


Goal and Approach
-----------------
Airbnb, Inc. is an online marketplace for arranging or offering lodging, primarily homestays, or tourism experiences. The company does not own any of the real estate listings, nor does it host events; it acts as a broker, receiving commissions from each booking.

The problem with airbnb is that it is hard to regulate. Anyone and everyone is allowed to be host and put up their listings to be booked into the airbnb site. With it comes the legitamacy of a listing. We have heard of stories of airbnb users being scammed one way or the other, from not being able to get a refund for a stay that did not go according to plan, or getting locked out of an apartment as the owners have no idea their apartment was listed on the website.
This can be a harrowing experience for airbnb users, and adds unncessary stress to an already stressful activity of trip planning.


My project has two parts.
1. We aims to explore the possibility of using machine learning to successfully classify potential suspicious listings. With this we can filter them out of the system so that they wont be "accidentally" chosen by unwary users.
2. With the updated listings, we can find similarity between each listings and provide effective recommendations for a user with their specific names

# Notebooks for Topic Modelling:

- [Classification Model 1](https://github.com/ikhwanwahid/Airbnb-Capstone-project-final/tree/master/LDA_model/Classification.ipynb)
- [Classification Model 2](https://github.com/ikhwanwahid/Airbnb-Capstone-project-final/tree/master/LDA_model/Classification2.ipynb)
- [Final Classification Model 2](https://github.com/ikhwanwahid/Airbnb-Capstone-project-final/tree/master/LDA_model/FinalClassification.ipynb)
- [December Classification Model](https://github.com/ikhwanwahid/Airbnb-Capstone-project-final/tree/master/LDA_model/DecemberClassification.ipynb)
- [Hypertuning](https://github.com/ikhwanwahid/Airbnb-Capstone-project-final/tree/master/LDA_model/Hypertuning.ipynb)


# Notebooks for Classification Modelling:

- [Filtering Listings](https://git.generalassemb.ly/DSI-SG-11/Ikhwan-Capstone-project-final/tree/master/LDA_model/Deployment_Test(1).ipynb)
- [Cleaning and Data Munging](https://git.generalassemb.ly/DSI-SG-11/Ikhwan-Capstone-project-final/tree/master/LDA_model/ListingsDatasetCleaning(1).ipynb)
- [Feature Engineering 1](https://git.generalassemb.ly/DSI-SG-11/Ikhwan-Capstone-project-final/tree/master/LDA_model/feature_engineering(2).ipynb)
- [Feature Engineering 2](https://git.generalassemb.ly/DSI-SG-11/Ikhwan-Capstone-project-final/tree/master/LDA_model/feature_engineering(3).ipynb)
- [PCA](https://git.generalassemb.ly/DSI-SG-11/Ikhwan-Capstone-project-final/tree/master/LDA_model/PCA(4).ipynb)
- [Doing K_means](https://git.generalassemb.ly/DSI-SG-11/Ikhwan-Capstone-project-final/tree/master/LDA_model/Doing_K_means(5).ipynb)


Dataset, Preprocessing, and Evaluation
--------------------------------------
My dataset was provided from the best website for airbnb scrapped data : https://http://insideairbnb.com/get-the-data.html. A total of 4 datasets was obtained from this website and they are as follows.

__NY December Listings__ 40000+ rows: http://data.insideairbnb.com/united-states/ny/new-york-city/2019-12-04/data/listings.csv.gz

__NY December Reviews__ 1000000+ rows: http://data.insideairbnb.com/united-states/ny/new-york-city/2019-12-04/data/reviews.csv.gz

__NY January Listings__ 40000+ rows : http://data.insideairbnb.com/united-states/ny/new-york-city/2019-01-09/data/listings.csv.gz

__NY January Reviews__ 800000+ rows: http://data.insideairbnb.com/united-states/ny/new-york-city/2019-01-09/data/reviews.csv.gz

Part 1: Topic Modelling
--------------

Preprocessing of the dataset was performed using Python. The preprocessing scripts are included in the LDA_model folder. In order to classify our listings into suspicious listings and legitamate ones, we will first be using the reviews datasets. These datasets had only 6 columns, but only 2 of them are of interest to us

| Interest | Non-interest |
|------|--------|
|listing_id|id|
|comments|date|
||reviewer_id|
||reviewer_name|

One of the first obstacles encountered was the absence of a target column. There is no column indicating if the listings in the datasets were suspicious in nature.
I made an informed decision to first compare listings found in December and January.
The number of listings that were missing in Dec from January was 25000+.
These listings were removed mainly because of two things.
<br>
<br>1.Owner has decided to not host anymore.
<br>2.Airbnb has removed them due to non-compliance to rules. (This is where our scams will be)


In a nutshell our text processing  process is as follows
 - Removing Stopwords
 - Creating bigrams and trigrams using gensim.models.phrases
 - Lemmatize the texts using spacey

Next we used the LDA model found in the gensim package to create our topics of interest.
<br>In our first iteration of the model, we picked a random number of topics to formulate.
That was 20 topics. After iteration this were the results.

![Results1](https://github.com/ikhwanwahid/Airbnb-Capstone-project-final/tree/master/LDA_model/Picture1.png)

The model does a good job in grouping words into the topics. We can more or less decipher based on the words in the topics what the topic is about. However what I discovered from the first iteration row_values

- 20 topics was too much, almost impossible to name all 20 topics.
- From the visualisation we can see that some topics are intertwined with each other, suggesting similarity between them
- We should decide on how many topics would serve our purpose, in this case (5)
  - Overall experience
  - Host experience
  - Host Communication
  - Location
  - Accomodation amenities

With that we repeat our model iteration again.
Subsequently through hypertuning we fitted a final model for our comments found in the December dataset.

![Results2](https://github.com/ikhwanwahid/Airbnb-Capstone-project-final/tree/master/LDA_model/Picture2.png)

What we have finally are 5 distinct topics modelled by LDA. Upon closer inspection, these topics obtained are in line with 5 topics that we have sought to achieve previously.

Finally I did a sentiment analysis onto the comments, To get an idea of the sentiment of each comments, and what the dominant topic that is associated with that comment

![Results3](https://github.com/ikhwanwahid/Airbnb-Capstone-project-final/tree/master/LDA_model/Picture3.png)



Part 2: Clustering
--------------

Preprocessing of the dataset was performed using Python. The preprocessing scripts are included in the December_final folder. The goal of part 2 was to recommend similar listings to a user based on the one he/she has already picked beforehand.

We have reduced the original columns of 107 columns down to 55. In the property_type columns, we see more then 20 different types of properties. We decided that the only listings that are of concern to us are those that already
  - Apartments
  - House
  - Townhouse
  - Condominiums

Subsequently we one hot encoded (get dummies) the categorical categories, and our final dataset contains 79 columns.

Principal component analysis was done on the features in our dataset, to obtain the non spatial relationships between the features. We got 50 components and subsequently did a K-means clustering to finally cluster the listings into 3  groups.

![Results4](https://github.com/ikhwanwahid/Airbnb-Capstone-project-final/tree/master/LDA_model/Picture4.png)

Suggestions for Future Work
---------------------------

Honestly my work here is mostly assumption work. I assume suspicious listings are once that contain negative comments belonging to 3 topic groups. While the end goal is to filter them out so that no users are able to choose them unknowingly, it would be of great help to actually get confirmation from DATA stating that these listings filtered out are indeed scams. By knowing the true postives ,true negatives of our model we can actually see how well it is performing. Subsequently we can retrain our model using other methods other then NLP, knowing there is a target value.
