# Capstone Project : Classify suspicious Airbnb Listings (NY)Improving overall UX !


Goal and Approach
-----------------
Airbnb, Inc. is an online marketplace for arranging or offering lodging, primarily homestays, or tourism experiences. The company does not own any of the real estate listings, nor does it host events; it acts as a broker, receiving commissions from each booking.

The problem with airbnb is that it is hard to regulate. Anyone and everyone is allowed to be host and put up their listings to be booked into the airbnb site. With it comes the legitamacy of a listing. We have heard of stories of airbnb users being scammed one way or the other, from not being able to get a refund for a stay that did not go according to plan, or getting locked out of an apartment as the owners have no idea their apartment was listed on the website.
This can be a harrowing experience for airbnb users, and adds unncessary stress to an already stressful activity of trip planning.


My project has two parts.
1. We aims to explore the possibility of using machine learning to successfully classify potential suspicious listings. With this we can filter them out of the system so that they wont be "accidentally" chosen by unwary users.
2. With the updated listings, we can find similarity between each listings and provide effective recommendations for a user with their specific names

# Notebooks for Topic Modelling:

- [Classification Model 1](https://git.generalassemb.ly/DSI-SG-11/Ikhwan-Capstone-project-final/tree/master/LDA_model/Classification.ipynb)
- [Classification Model 2](https://git.generalassemb.ly/DSI-SG-11/Ikhwan-Capstone-project-final/tree/master/LDA_model/Classification2.ipynb)
- [Final Classification Model 2](https://git.generalassemb.ly/DSI-SG-11/Ikhwan-Capstone-project-final/tree/master/LDA_model/FinalClassification.ipynb)
- [December Classification Model](https://git.generalassemb.ly/DSI-SG-11/Ikhwan-Capstone-project-final/tree/master/LDA_model/DecemberClassification.ipynb)
- [Hypertuning](https://git.generalassemb.ly/DSI-SG-11/Ikhwan-Capstone-project-final/tree/master/LDA_model/Hypertuning.ipynb)


# Notebooks for Classification Modelling:

- [Filtering Listings](Deployment_Test(1).ipynb)
- [Cleaning and Data Munging](ListingsDatasetCleaning(1).ipynb)
- [Feature Engineering 1](feature_engineering(2).ipynb)
- [Feature Engineering 2](feature_engineering(3).ipynb)
- [PCA](PCA(4).ipynb)
- [Doing K_means](Doing_K_means(5).ipynb)


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
1.Owner has decided to not host anymore.
2.Airbnb has removed them due to non-compliance to rules. (This is where our scams will be)





| Basic | Advanced | Deep|
|------|--------|-------|
|latitude|review_scores_rating|description|
|longitude|reviews_per_month|picture_url|
|property_type|number_of_reviews|host_picture_url|
|room_type|host_response_time
|accommodates|host_response_rate
|bathrooms|host_is_superhost
|bedrooms|host_total_listings_count
|beds|host_verifications
|cancellation_policy|host_identity_verified
|bed_type|host_acceptance_rate
|amenities|
|guests_included|
|minimum_nights|


Basic Analysis
--------------
Once our data was preprocessed, we included the basic features from Table 2, and ran multiple different regression techniques (note, we did not include features with a * in the table, and choose a 3,000 listing subset of NYC data). We choose to run Linear, Lasso, Ridge, and Elastinet standard regressions to help with different dependencies that may be present within the data, along with a random forest and a gradient boosted regressor to represent ensemble methods. Our initial results are shown in Table 3. Looking at these values, we were surprised that our standard regression techniques were performing much better than our ensembles on test evaluation. We dove into the data and discovered two key insights. First, thinking about how we would personally search for an Airbnb, we realized that location was likely the most important feature. However, when we looked at our basic attributes we realized that longitude and latitude were the only features representing location - and our model may have not been powerful enough to truly utilize this complex representation. We revisited the available features and added in the neighborhood feature - which is a location based filter provided to users querying for an airbnb listing. Additionally, we looked further into the New York dataset, as removing the tuples from our initial approach had significantly impacted our scores, as shown in the right hand side of Table 3. When visualizing our data with Tableau (shown in Figure 1, we realized that the New York dataset was not fully representing the market diversity well, as the dataset was not fully shuffled. Shuffling our New York dataset (shown in Figure 2) and adding in location - our train and test accuracy increased substantially as shown in Table 4. This accuracy bump represented a core feature in machine learning that we discussed in class around the Netflix recommendation prize - relevant features with a simple model are significantly more important than extraneous features with a very complex model.  

From here, we tried tuning the hyperparameters of each model, largely being unsuccessful. We again turned to examining our overall approach. We noticed that when modifying our parameters, there were signs of overfitting. Additionally, we dove into the actual prices of Airbnb’s within each city and noticed that there were some listings that had an astronomical price difference (over $700 per night) - particularly in New York. Figure 3 represents the distributions. To alleviate these issues we put a variance limit on features (features under a variance limit would not be included) and a price limit on listings to be included. This pushed our model scores even higher - and they can be found in Table 5.


Suggestions for Future Work
---------------------------
Our conclusions to our work are summarized at a high level on the main landing webpage. In terms of directing future work, there are many things that we would like to do. First, we would very much like to see how our model generalizes to other cities (with some training data). Also - we think trying to extract information from the images included could potentially be very powerful.

Group Member Work
-----------------
Throughout the duration of the project, all group members put substantial work into the direction and execution of the work completed. However, segmenting the work as requested,  Albert focused on exploratory feature analysis, Keith was responsible for model and feature selection, and Rhett and Lukas built, trained and tuned the models. We feel strongly that each member was essential and had great value add.

![Results1](https://github.com/Lukas-Justen/Airbnb-Price-Evaluator/raw/master/docs/img/results1.png)
![Results2](https://github.com/Lukas-Justen/Airbnb-Price-Evaluator/raw/master/docs/img/results2.png)
![NewYork](https://github.com/Lukas-Justen/Airbnb-Price-Evaluator/raw/master/docs/img/ny.png)
![Prices](https://github.com/Lukas-Justen/Airbnb-Price-Evaluator/raw/master/docs/img/prices.png)

_by Keith Pallo, Rhett Dsouza, Albert Z. Guo, Lukas Justen_
