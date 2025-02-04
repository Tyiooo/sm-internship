import pandas as pd
import random

tasks_to_do = [
    "programming",
    "watch cinema",
    "play games",
    "write a diploma",
    "sleep",
    "go to the gym",
    "play guitar",
    "go out with friends",
]


def main():
    print("Hello from sm-internship!")


if __name__ == "__main__":
    print(tasks_to_do[random.randint(0, len(tasks_to_do) - 1)])
    print(pd.__version__)
    main()
