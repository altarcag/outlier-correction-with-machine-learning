the temperature dataset contains some outliers which clearly are caused by instrumental error. There were many days with a value of -72 degrees celsius in weather dataset. all the outliers had the same -72 value. So this was clearly an instrumental or any other kind of error.

I used machine learning algorithms to make predictions for the entire dataframe, and then replaced the days with an outlier with the predictions.
this same algo will then be used for a ground temperature dataset which has missing days due to sensors not being able to record on cloudy days.
