from atproto import Client
import json
import resnext as resnext
import os
import random

pwd_f = "pswd.json"
source_img_folder = "./imgs/submissions/"
done_img_folder = "./imgs/posted/"

# follows example from
# https://github.com/MarshalX/atproto/blob/main/examples/send_images.py
def post_single_image(img_f, text) -> None:
    login = json.load(open(pwd_f))

    client = Client()
    client.login(login['uname'], login['pwd'])
    imgs = [ open(img_f, 'rb').read() ]
    client.send_images(text=text, images=imgs)

def gen_tweet_string(prediction, img_path):
    tweet_str =  "top 5 guesses:\n"
    for i in range(5):
        tweet_str +=  "%s [ %.3f ]\n"%(prediction[0][i], prediction[1][i])
    return tweet_str

def main() -> None:
    #1. Grab a random image
    img_list = os.listdir(source_img_folder)
    img_f = random.choice(img_list)
    img_path = source_img_folder + img_f

    #2. Run it through resnext
    prediction = resnext.resnext_classify(img_path)

    #3. Generate the tweet string
    text = gen_tweet_string(prediction, img_path)
    print(text)

    #4. Post the image
    post_single_image(img_path, text)

    #5. Move the image to the posted folder
    os.rename(img_path, done_img_folder + img_f)

if __name__ == '__main__':
    main()