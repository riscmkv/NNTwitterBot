import twython as Twython
import NNClassify
import resnext
import confidence
import time

def postTweet(fanSubmit=False):
    twitter = Twython(
        'consumer_key',
        'consumer_secret',
        'access_token',
        'access_token_secret'
    )

    img_data = NNClassify.GetImage(fanSubmit)
    img_path = img_data[0]
    submitter = img_data[1]

    NNClassify.logger.info('----------------------------')
    NNClassify.logger.info('new tweet being sent...')
    NNClassify.logger.info('time: ' + str(time.time()))
    NNClassify.logger.info('img path: ' + img_path)
    NNClassify.logger.info('submitter: ' + submitter)

    prediction = resnext.resnext_classify(img_path)
    message = NNClassify.gen_tweet_string(prediction, img_path)

    if(submitter != 'none') and (fanSubmit):
        message = message + '\nSubmission by @.' + submitter.replace('@', '')

    NNClassify.logger.info('message: ' + message)

    img_post = open(img_path, 'rb')
    response = twitter.upload_media(media=img_post)
    media_id = [response['media_id']]
 
    try:
        twitter.update_status(status=message, media_ids=media_id)
    except:
        NNClassify.logger.info("update failed! Could not connect?")
        return
    
    img_post.close()

    if(not NNClassify.QueueIsEmpty()):
        NNClassify.logger.info('removing from queue...')
        NNClassify.RemoveFromQueue(img_path)
	
    #NNClassify.logger.info('checking submissions...')
    #grab_submissions()

	
def PostTweetFname(fname, customMsg=None, hideSubmitter=False):
    twitter = Twython(
        'consumer_key',
        'consumer_secret',
        'access_token',
        'access_token_secret'
    )

    img_path = fname
    submitter = NNClassify.getSubmitterName(fname)

    prediction = resnext.resnext_classify(img_path)
    message = NNClassify.gen_tweet_string(prediction, img_path)
    
    if customMsg:
        message = message + "\n" + customMsg

    if(submitter != 'none') and (not hideSubmitter):
        message = message + '\nSubmission by @.' + submitter

    img_post = open(img_path, 'rb')
    response = twitter.upload_media(media=img_post)
    media_id = [response['media_id']]
    twitter.update_status(status=message, media_ids=media_id)
    img_post.close()

    if(not NNClassify.QueueIsEmpty()):
        NNClassify.logger.info('removing from queue...')
        NNClassify.RemoveFromQueue(img_path)