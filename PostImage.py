import argparse
import twitterBotAPI

def main():
    parser = argparse.ArgumentParser(description="Posts a randomly drawn image",
                            add_help=True)
    args = parser.parse_args()

    twitterBotAPI.postTweet()

if __name__ == "__main__":
    main()