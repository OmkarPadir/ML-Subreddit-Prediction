# ML-Subreddit-Prediction

# Main code files
1] community-detector-scrapper.py : Download data from reddit api, store thumbnails in images folder and store posts titles in submissions.csv. <br>
2] ImageModel.py : CNN image model for subreddit prediction <br>
3] train_text_models.py : Contains text preprocessing, two word representation techniques. LR and KNN models are trained on the text features. <br>
4] combine_models.py : Loads the trained text and image models, combines their features and trains a third LR model. <br> 

# Supporting code files <br>
1] GetImageNamesinCSV : List of downloaded images <br>
2] Rename Images.py : Rename images to avoid overwriting due to same names <br>
3] SplitFiles.py : Split images in Train and Test Set <br>
