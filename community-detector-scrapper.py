import requests
import pandas as pd
import urllib.request as url
import datetime as dt
import time
import re
import os

reddit_submissions = []

dest ='images'
if not os.path.exists(dest):
    os.makedirs(dest)

def api_call(subreddit, after, before):
    base_url = 'https://api.pushshift.io/reddit/search/submission/'
    payload = {'subreddit': str(subreddit), 'after': str(after), 'before': str(before), 'sort': 'desc', 'limit': 1000}
    r = requests.get(base_url, params=payload)
    response_body = r.json()
    print(after, before, len(response_body['data']))
    for submissions in response_body['data']:
        reddit_submissions_title_url_pair = [subreddit, submissions['title'], submissions['thumbnail']]
        reddit_submissions.append(reddit_submissions_title_url_pair)


before_epochs = []
after_epochs = []
for i in range(1, 12):
    for j in range(1, 26, 2):
        before = int(dt.datetime(2019, i, j + 2, 0, 0).timestamp())
        after = int(dt.datetime(2019, i, j, 0, 0).timestamp())
        before_epochs.append(before)
        after_epochs.append(after)

print(len(before_epochs))
print(before_epochs)
print(after_epochs)

categories = ["food", "sports", "travel", "technology", "science"]
for subreddit in categories:
    for i in range(len(before_epochs)):
        api_call(subreddit, after_epochs[i], before_epochs[i])
        time.sleep(3)

print(len(reddit_submissions))

df = pd.DataFrame(reddit_submissions, columns=['subreddit', 'reddit_submissions_title', 'reddit_submissions_image_urls'])
df.to_csv('./submissions_titles.csv', header=True, index=False, columns=list(df.axes[1]))
print(df)

for ind in df.index:
    print(str(df['reddit_submissions_image_urls'][ind]), ind)
    image_url = str(df['reddit_submissions_image_urls'][ind])
    # if re.search("^https://imgur.com/.*", image_url):
    #     image_url = image_url[:8] + "i." + image_url[8:] + ".jpg"
    #     df['reddit_submissions_image_urls'][ind] = image_url
    #     print("new url->", df['reddit_submissions_image_urls'][ind])
    try:
        url.urlretrieve(str(df['reddit_submissions_image_urls'][ind]), "./images/" + str(df['subreddit'][ind]) + "-" + str(ind) + ".jpg")
    except:
        continue
