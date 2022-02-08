import random
import json
from urllib import request
import shutil
import matplotlib.pyplot as plt
import pickle
from twython import Twython
import glob
import time
import string
import oauth2 as oauth
import logging
import resnext
import confidence

logger = logging.getLogger('image bot')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('imbot.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

from auth import (
    consumer_key,
    consumer_secret,
    access_token,
    access_token_secret
)

twitter = Twython(
	consumer_key,
	consumer_secret,
	access_token,
	access_token_secret
)


img_ext = ['jpg', 'png', 'jpeg']
CIredditURL = 'http://reddit.com/r/cursedimages/.json'

def isImageURL(post):
    for ext in img_ext:
        if(post['data']['url'].endswith(ext)):
            return True
    return False

def isOver18URL(post):
    return post['data']['over_18']

def getRedditJSON(redditURL):
    try:
        f = request.urlopen(redditURL)
    except:
        print("Could not open reddit json! perhaps too many requests...")
    else:
        json_dat = bytes.decode(f.read())
        json_dat = json.loads(json_dat)
        pickle.dump(json_dat, open('reddit_info.json', 'wb'))

def GetRandomImage():
    json_dat = pickle.load(open('reddit_info.json', 'rb'))

    imgurl = []
    posts = json_dat['data']['children']
    for post in posts:
        if((not isOver18URL(post)) and isImageURL(post)):
            imgurl.append(post['data']['url'])

    rand_url = random.sample(imgurl, 1)[0]
    request.urlretrieve(rand_url, './random_img/temp_img.' + rand_url.split('.')[-1])
    return './random_img/temp_img.' + rand_url.split('.')[-1]

def getSubmitterName(fname):
    if(len(fname.split('.')) < 3):
        return 'none'
    else:
        return fname.split('.')[-2]

def QueueIsEmpty():
    submissions = glob.glob("./submissions/*")
    curated = glob.glob("./curated/*")
    
    if (len(submissions) != 0) or (len(curated) != 0):
        return False
    else:
        return True

def GetImage(fanSubmit=False):
    submissions = glob.glob("./submissions/*")
    if fanSubmit:
        submissions = glob.glob("./fan_submit/*")
    curated = glob.glob("./curated/*")
    
    if(len(submissions) != 0):
        img_path = random.sample(submissions, 1)[0]
        submitter = getSubmitterName(img_path)
        return [img_path, submitter]
    elif(len(curated) != 0):
        img_path = random.sample(curated, 1)[0]
        return [img_path, 'none']
    else:
        getRedditJSON(CIredditURL)
        getRedditJSON(CIredditURL)
        return [GetRandomImage(), 'none']
    
def RemoveFromQueue(img_path):
    fname = img_path.split('\\')[-1]
    shutil.move(img_path, "./submissions_completed/")

def getSubmitterName(fname):
    if(len(fname.split('.')) < 3):
        return 'none'
    else:
        return fname.split('.')[-2]
	
def get_message_response_no_pagination(**kwargs):
    twitter_response = twitter.request(
        'https://api.twitter.com/1.1/direct_messages/events/list.json',
        method='GET',
        params=kwargs,
        version='1.1'
    )
    return twitter_response

def get_messages(threshold=200):
    messages = []
    response = get_message_response_no_pagination()
    messages = response['events']
    while len(messages) < threshold:
        nc = response['next_cursor']
        response = get_message_response_no_pagination(cursor=nc)
        messages = messages + response['events']
    return messages
    
def list_saved_images(folders):
    files = []
    for folder in folders:
        files = files + glob.glob("./" + folder + "/*")
    for i in range(len(files)):
        files[i] = files[i].split('\\')[-1]
    return files

def get_screen_name(userid, twitter):
    uname_response = twitter.request(
        'https://api.twitter.com/1.1/users/show.json',
        method='GET',
        params={'user_id' : userid},
        version='1.1'
    )
    return uname_response['screen_name']

def is_in_list(needle, haystack):
    for straw in haystack:
        if needle == straw:
            return True
    return False

def send_direct_message(userid, msg, twitter):
    twitter_response = twitter.request(
        'https://api.twitter.com/1.1/direct_messages/events/new.json',
        method='POST',
        params='{"event": {"type": "message_create", '
                '"message_create": {"target": {"recipient_id": "'+userid+'"}, '
                '"message_data": {"text": "'+msg+'"}}}}',
        version='1.1'
    )
    return twitter_response

def genRandomString(length):
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for i in range(length))    

folders = ['submissions', 'submissions_completed', 'submissions_rejected', 'submissions_unmoderated']

def grab_submissions():
    twitter = Twython(
        consumer_key,
        consumer_secret,
        access_token,
        access_token_secret
    )

    # 1. Grab all new messages
    msgs_dict = get_messages()
    # 2. Iterate through the messages
    for message in msgs_dict:
        # 3. Check if the message has a image attached
        if 'attachment' in message['message_create']['message_data'] :
            # 4. If it does, get the name of the image
            img_url = message['message_create']['message_data']['attachment']['media']['media_url']
            img_name = img_url.split('/')[-1]
            img_sender_id = message['message_create']['sender_id']
            img_sender_sn = get_screen_name(img_sender_id, twitter)
            img_name = img_name.split('.')[-2] + '.' + img_sender_sn + '.' + img_name.split('.')[-1]
            # 5. Check if the image already exists in one of the folders
            saved_images = list_saved_images(folders)
            #print(img_name, is_in_list(img_name, saved_images))
            if not is_in_list(img_name, saved_images):
                # 6. If it does not, download the image

                logger.info('found new submission...')
                logger.info('submitter: ' + img_sender_sn)
                logger.info('img_name: ' + img_name)

                consumer2 = oauth.Consumer(key=consumer_key, secret=consumer_secret)
                token2 = oauth.Token(key=access_token, secret=access_token_secret)
                client = oauth.Client(consumer2, token2)
                response, data = client.request(img_url)

                #print(type(data))
                f = open('./submissions_unmoderated/' + img_name, 'wb')
                f.write(data)
                f.close()

                # 7. Send a message to the sender that his image is being moderated
                submission_message = "Automated Message: This image has been sent for moderation. If approved, it will be added to the queue!"
                send_direct_message(img_sender_id, submission_message, twitter)
		
                logger.info('message sent!')
            else:
                continue
        else:
            continue

def is_boring_prediction(prediction):
    prediction_text = prediction[0][0]
    return (prediction_text == 'comic book') or (prediction_text == 'book jacket') or (prediction_text == 'web site')

def gen_tweet_string(prediction, img_path):
    message = None
    if is_boring_prediction(prediction):

        confidence_calced = [None, None, None, None]
        confidence_calced[0] = confidence.calc_confidence_idx(img_path, prediction[2][0])
        confidence_calced[1] = confidence.calc_confidence_idx(img_path, prediction[2][1])
        confidence_calced[2] = confidence.calc_confidence_idx(img_path, prediction[2][2])
        confidence_calced[3] = confidence.calc_confidence_idx(img_path, prediction[2][3])

        message = "Image prediction:\n"
        message += prediction[0][0] + " (" + str(round(confidence_calced[0]*100, 2)) + "%)\n"
        message += prediction[0][1] + " (" + str(round(confidence_calced[1]*100, 2)) + "%)\n"
        message += prediction[0][2] + " (" + str(round(confidence_calced[2]*100, 2)) + "%)\n"
        message += prediction[0][3] + " (" + str(round(confidence_calced[3]*100, 2)) + "%)"

    else:
        (certainty, img_classification) = (confidence.calc_confidence_idx(img_path, prediction[2][0]), prediction[0][0])
        message = "Image prediction: " + img_classification + "\n" + "Confidence: " + str(round(certainty*100, 2)) + "%"
    return message

def gen_additional_prediction_string(prediction, img_path):
    confidence_calcs = [None, None, None]
    confidence_calcs[0] = confidence.calc_confidence_idx(img_path, prediction[2][1])
    confidence_calcs[1] = confidence.calc_confidence_idx(img_path, prediction[2][2])
    confidence_calcs[2] = confidence.calc_confidence_idx(img_path, prediction[2][3])
    message = 'additional image predictions:\n'
    message += '- ' + prediction[0][1] + ' (confidence: ' + str(round(confidence_calced[0]*100, 2)) + '%)\n'
    message += '- ' + prediction[0][2] + ' (confidence: ' + str(round(confidence_calced[1]*100, 2)) + '%)\n'
    message += '- ' + prediction[0][3] + ' (confidence: ' + str(round(confidence_calced[2]*100, 2)) + '%)\n'
    return message

def peek_prediction(img_path):
    prediction = resnext.resnext_classify(img_path)
    conf = str(round(confidence.calc_confidence_idx(img_path, prediction[2][0]), 2))
    print(prediction[0][0] + " " + conf)

def postTweet(fanSubmit=False):
    twitter = Twython(
        consumer_key,
        consumer_secret,
        access_token,
        access_token_secret
    )

    img_data = GetImage(fanSubmit)
    img_path = img_data[0]
    submitter = img_data[1]

    logger.info('----------------------------')
    logger.info('new tweet being sent...')
    logger.info('time: ' + str(time.time()))
    logger.info('img path: ' + img_path)
    logger.info('submitter: ' + submitter)

    prediction = resnext.resnext_classify(img_path)
    message = gen_tweet_string(prediction, img_path)
    additional_message = gen_additional_prediction_string(prediction, img_path)

    if(submitter != 'none') and (fanSubmit):
        message = message + '\nSubmission by @' + submitter

    logger.info('message: ' + message)

    img_post = open(img_path, 'rb')
    response = twitter.upload_media(media=img_post)
    media_id = [response['media_id']]
 
    try:
        status = twitter.update_status(status=message, media_ids=media_id)
        if not is_boring_prediction(prediction):
            twitter.update_status(status=additional_message, in_reply_to_status_id=status.id_str)
    except:
        logger.info("update failed! Could not connect?")
        return
    
    img_post.close()

    if(not QueueIsEmpty()):
        logger.info('removing from queue...')
        RemoveFromQueue(img_path)
	
    #logger.info('checking submissions...')
    #grab_submissions()

	
def PostTweetFname(fname, customMsg=None):
    twitter = Twython(
        consumer_key,
        consumer_secret,
        access_token,
        access_token_secret
    )

    img_path = fname
    submitter = getSubmitterName(fname)

    prediction = resnext.resnext_classify(img_path)
    message = gen_tweet_string(prediction, img_path)
    additional_message = gen_additional_prediction_string(prediction, img_path)
    
    if customMsg:
        message = message + "\n" + customMsg

    if(submitter != 'none'):
        message = message + '\nSubmission by @' + submitter

    img_post = open(img_path, 'rb')
    response = twitter.upload_media(media=img_post)
    media_id = [response['media_id']]
    status = twitter.update_status(status=message, media_ids=media_id)
    if not is_boring_prediction(prediction):
        twitter.update_status(status=additional_message, in_reply_to_status_id=status.id_str)
    img_post.close()

    if(not QueueIsEmpty()):
        logger.info('removing from queue...')
        RemoveFromQueue(img_path)
