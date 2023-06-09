# ML Model to Predict Position of Football Players - Serie A

## Objective

The idea behind this Data Analysis Project is to predict the position of a football player using the statistics of players playing in "Serie A".

My project collects data from FBREF.com, using web scrapping techniques to obtain statistic regarding "Serie A"'s players. Then I selected only the players that have played more the 1000 minutes in the last season. Among all the available statistics, I chose the 25 more significant and I used them to build a Neural Network that is able to predict the position of players with an accuracy of 87%. 

## First phase - Web Scrapping and creation of the Database

The first step of my project was the creation of the Database. I use the library "Beautiful Soup" to parse the html page and gather the statistics. The using the library "Pandas" I created the dataframe and exported it as a xlsx file. The code is projected to scrap the data only one time.

## Second phase - Creating the Machine Learing model

I decided to use a Neural Network to classify the players into four categories: Goalkeeper (GK), Defender (DF), Midfielder (MF), Forward (FW). 
The 25 statistics were organized in a 5x5x1 image. The Neural Network is built with 6 hidden layer: three Dense layers, two Dropout layers and a Flatten layer. I used two additional Dense layers as input/output layer. 

I split the dataset, using a 0.4 ratio, into a training set and an evaluating set. 

## Possible improvements

The project can be improved in several ways. The database can be expanded including players from different national leagues and see if the model can predict correctly the player's position. We can refine the classification including more positions and analyze the model's behaviour. Another possible improvement is the enlargment of the set of statistics used to predict the position and analyze different combinations.

