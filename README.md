Web App hosted here: https://precog-task.herokuapp.com

# About the project     
1. Collect a minimum of 300 tweets (excluding retweets/replies) from 5 verified police accounts (India).
2. Store the collected data in MongoDB Collections.
3. Perform statistical analysis on the data.
4. Display the results in the form of a web app.

# Solution
## Approach
1. Data Collection: `Selenium` & `BeautifulSoup` were used to scrape the web pages of the required twitter accounts.  
2. Clean Up & Storage: Collected data was cleaned & stored in `MongoDB` collections hosted on [mLab](https://mlab.com). It can be accessed at `mongodb://tweets_database:2222@ds145868.mlab.com:45868/tweets_database`
3. Analysis: Analysis was purely done in iPython Notebook and I'd like to request you to please check them out `(Analysis.ipynb)` once since they provide a very conducive environment for data analysis.
4. Hosting: Used simple flask based web app to display the results, deployed on heroku. [Click Here](https://precog-task.herokuapp.com)

## Primary areas of focus
1. Simplicity of results with maximum information gain.
2. Easily understandable code.

## Details
### Data Collection
1. Exclude Retweets: To overcome this, the scrapper (bs4) was made to filter tweets on the basis of `data-user-id` which is unique for any given tweet and always traces back to the original source of the tweet.
2. Exclude Replies: URLs taken were of the form `https://twitter.com/DelhiPolice` and **NOT** `https://twitter.com/DelhiPolice/with_replies`.
3. Why create five different scripts to collect data? It's true that this could've been accomplished in a single script however, given the size and nature of data and the constant debugging that was required during development (collection), I felt it was better to create them individually. Also, if we wanted to update a collection of a particluar account, the need to collect data from all the other ones did not make much sense. (But this is an extremely personal opinion.) Executing any of the five scripts will alter its respective collection stored in the database automatically (online), thus accomodating the changes.

### Analysis
`Analysis.ipynb` : All the analysis was done in an `IPython Notebook`. There were 2 reasons to use iPy notebooks - Firstly, they provide a rich environment that combines module by module code execution, mathematics & plots. Second and more importantly, that's what I've been using for a quite a while now for my Kaggle Competitions/Projects. 

1. **Frequency of Tweets (tweets/day)**: Includes the average number of tweets made by the account (no rts/replies). 
2. **Frequent Hashtags (#)**: Includes 10 most frequently used hashtags.
3. **Sentiment Analysis**: Done using `TextBlob`. Pie chart denotes the total number of positive, negative & neutral tweets. 
4. **Engagements**: Determining the avg number of engagements (Favs + RTs), grouped by the type of content (Text/Media)
5. **Time Series**: Time series is important (I feel so) when we want to study the activity of such a law enforcement social handle under different instances of time. Therefore, I think it was important to include it in the statistical analysis part.
6. **Word Cloud**: Word clouds have been long used to represent frequently occuring words. Additionally to that, the wordclouds generated in this analysis mask the image of the state/city's MAP itself. (Except- Thane, where I could not find a decent png map image). 

I would request you to please checkout the IPython Notebook since it provides ease of reading the code and it can be executed there & then module by module (please execute the relevant 'initial' modules before proceeding on to the analytical plots)

### Directory/Files Description
1. `app.py` : For Flask deployment. 
2. `Analysis.py` : Analysis of the collected data.
3. `/FetchData/` : Directory containing data collection scripts. One may comment out the last few lines of code in these scripts to avoid altering the collections conatining the original data, they'll still execute and print the result. (**Note : Please change the path to `chromedriver` while executing**)
4. `/static/` : Contanis `style.css` and other relevant files necessary for deployment.
5. `/templates/` : Contains the `hello.html` template.
6. `/JSON_Files/` : Contains JSON exports for all the accounts from MongoDB Collections

### Records
1. @DelhiPolice - 358 Records
2. @MumbaiPolice - 617 Records
3. @PuneCityPolice - 347 Records
4. @ThaneCityPolice - 313 Records
5. @wbpolice - 393 Records

### Tools/Libraries/Env Used
1. python3 (spyder-conda)
2. IPython Notebook (For analysis)
3. NumPy/Pandas/PyMongo etc (Essentially reqd)
4. Matplotlib, Seaborn, WordCloud for visualizations
5. NLTK 
6. TextBlob (Sentiment Analysis)
7. Selenium/BS4 

## Challenges/End Note
One of the key challenges for me was to work out the web-deployment solution since I was just starting out with the python based web frameworks, so that took around an entire day or two. Another one, though straight forward, was to bypass the limitations posed by the twitter REST APIs (3200 tweets/filtering of RTs etc) for which I did try to work for a day or so (on the APIs), looking for solutions but ultimately went ahead with the web-scrapping based solution for data collection.

Further work:
1. Structure the code in a better way(there might be some redundancy here & there).
2. Apart from the specific task at hand, I'd like to explore the possibility of using machine learning based techniques like cluster analysis along some supervised models to identify specific patterns of factors influencing engagements etc. (Just a thought) 
3. Make the solution more dynamic in nature in terms of deployment.
