import json
import os


def Configuration():
    current_dir = os.path.dirname(__file__)

    file_path = os.path.join(current_dir, "config.json")

    file_path = os.path.abspath(file_path)
    fileName = file_path
    try:
        with open(fileName, 'r') as file:
            return json.load(file)
    except:
        print("Error on read config file")
        exit()
