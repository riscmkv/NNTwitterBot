import argparse
import twitterBotAPI

def main():
    parser = argparse.ArgumentParser(description="Posts a tweet with a given filename with a custom name",
                            add_help=True)
    parser.add_argument('--fname', dest="fname", type=str, help="filename")
    parser.add_argument('--custommsg', dest="custommsg", type=str, help="custom message")
    args = parser.parse_args()

    twitterBotAPI.PostTweetFname(args.fname, customMsg=args.custommsg)

if __name__ == "__main__":
    main()