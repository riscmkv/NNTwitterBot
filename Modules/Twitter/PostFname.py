import argparse
import twitterBotAPI

def main():
    parser = argparse.ArgumentParser(description="Posts a tweet with a given filename with a custom name",
                            add_help=True)
    parser.add_argument('--fname', dest="fname", type=str, help="filename")
    parser.add_argument('--custommsg', dest="custommsg", type=str, help="custom message")
    parser.add_argument('--hide-submitter', dest="hidesubmitter", type=bool, help="Hide name of submitter")
    parser.add_argument('--peek-prediction', dest="peekprediction", type=str, help="Peek the prediction of an image")
    args = parser.parse_args()

    if (args.peekprediction is None):
        twitterBotAPI.PostTweetFname(args.fname, customMsg=args.custommsg, hideSubmitter=args.hidesubmitter)
    else:
        twitterBotAPI.peek_prediction(args.peekprediction)

if __name__ == "__main__":
    main()