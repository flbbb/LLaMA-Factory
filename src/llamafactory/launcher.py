from dotenv import load_dotenv

from llamafactory.train.tuner import run_exp


def launch():
    load_dotenv()
    run_exp()


if __name__ == "__main__":
    launch()
