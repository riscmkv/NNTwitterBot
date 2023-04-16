import argparse
import twitterBotAPI

def main():
    parser = argparse.ArgumentParser(description="Posts a randomly drawn image",
                            add_help=True)
    parser.add_argument('--fansubmit', dest="fansubmit", type=bool,
                        help="Whether or not to draw image from submission pool")
    args = parser.parse_args()

    twitterBotAPI.postTweet(fanSubmit=args.fansubmit)

if __name__ == "__main__":
    main()