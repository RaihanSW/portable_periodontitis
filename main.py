from app.utils import InputDetection
from app.function.func import Detection

import os
from pathlib import Path
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive

quit = 0

print(
    "Welcome to Periodontitis Detector\n"
    "Python program for detecting periodontitis using ML\n"
    "\n"
    "Made by - UGM Joint Research for Periodontitis\n"
    "ver 0.1"
)


def main():
    global quit
    print(
        "\n"
        "We have 5 kind of detection :\n"
        "1. Boneloss detection\n"
        "2. Lower CEJ detection\n"
        "3. Lower Teeth(prob) detection\n"
        "4. Upper CEJ detection\n"
        "5. Upper Teeth(prob) detection\n"
    )

    option = input("Choose 1-5 for detection option :")
    choice = InputDetection(option)
    if choice == "quit":
        quit = 1
    elif choice:
        # gauth = GoogleAuth()
        # gauth.LocalWebserverAuth()
        # drive = GoogleDrive(gauth)
        insert_dir = Path("./insert_here")
        image = os.listdir(insert_dir)[0]
        Detection(image,choice)


while quit == 0:
    main()
