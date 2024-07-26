# ROSS.1RPS

 ROSS, a future AI system driven by machine learning and powered by an nvidia jetson nano. In the future ROSS will become a flight computer and will act as a one-man Mission Control system. At the moment in this very rough version of ROSS, it is capable of playing rock paper scissors. Later on it will get some updates to allow it to work its way up and become a possible future in the aerospace field. >  

![add image descrition here](direct image link here)

### The Algorithm
The Algorithm this project runs on uses AI to recognize images. Using specific details obtained through hours of training, the computer is able to narrow down what kind of hand gesture (rock, paper, or scissors) it is looking at.

### Running this project
1. **Open VS Code and connect to your Jetson Nano.**
2. **Make sure you have installed Jetson Inference and Docker Image from:** 
   [Jetson Inference Installation](https://github.com/dusty-nv/jetson-inference/blob/master/docks/building-repo-2.md)
3. **Download the file and drag it into your library on the device where you're running the program. Make sure it's located in the `jetson-inference/python/training/classification/data`.**
4. **"cd" to `nvidia/jetson-inference/`.**
5. **Once it has run and you're back in the `jetson-inference` folder, run `./docker/run.sh` to run the docker container. You may need to re-enter your NVIDIA password at this step.**
6. **From inside the Docker container, change directories so you are in `jetson-inference/python/training/classification`.**
7. **Run the training script to re-train the network where the model-dir argument is where the model should be saved and where the data is:**
   ```bash
   python3 train.py --model-dir=models/RPS data/RPS
   ```
8. **You should immediately start to see output, but it will take a very long time to finish running. It could take hours depending on how many epochs you run for your model.**
9. **When running the model, you can also specify the value of how many epochs and batch sizes you want to run. For example, at the end of that code you can add:**
   ```bash
   --batch-size=NumberOfBatchFiles --workers=NumberOfWorkers --epochs=NumberOfEpochs
   ```
10. **While it's running, you can stop it at any time using `Ctrl+C`. You can also restart the training again later using the `--resume` and `--epoch-start` flags, so you don't need to wait for training to complete before testing out the model.**
11. **Make sure you are in the docker container and in `jetson-inference/python/training/classification`.**
12. **Run the ONNX export script:**
    ```bash
    python3 onnx_export.py --model-dir=models/RPS
    ```
13. **Look in the `jetson-inference/python/training/classification/models/RPS` folder to see if there is a new model called `resnet18.onnx` there. That is your re-trained model.**
14. **Exit the docker container by pressing `Ctrl + D`.**
15. **On your Nano, navigate to the `jetson-inference/python/training/classification` directory.**
16. **Use `ls models/RPS/` to make sure that the model is on the Nano. You should see a file called `resnet18.onnx`.**
17. **Set the NET and DATASET variables:**
    ```bash
    NET=models/RPS DATASET=data/RPS
    ```
18. **Run this command to see how it operates on an image from the test folder:**
    ```bash
    imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/name/name.jpg TestImage1.jpg
    ```
    [View a video explanation here](https://www.youtube.com/watch?v=l_iidyeYtJM)

**Link to Rock-Paper-Scissors Dataset:** [Rock-Paper-Scissors Dataset](https://www.kaggle.com/datasets/glushko/rock-paper-scissors-dataset)

### Code
```python
import cv2
import numpy as np
import random
import time
import os

# Load the Rock-Paper-Scissors dataset
rock_paper_scissors_dataset = ...

# Define the camera capture
cap = cv2.VideoCapture(0)

# Define the game logic
def play_game():
    print("Let's play Rock-Paper-Scissors!")
    rounds = 3
    player_wins = 0
    ross_wins = 0

    for i in range(rounds):
        print(f"Round {i+1}...")
        time.sleep(3)  # 3-second timer

        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error capturing frame!")
            break

        # Preprocess the frame
        frame = cv2.resize(frame, (224, 224))
        input('Type return when ready. ')
        # Recognize the hand gesture
        hand_gesture = recognize_hand_gesture(frame, rock_paper_scissors_dataset)

        # Generate ROSS's response
        ross_response = random.choice(["rock", "paper", "scissors"])
        print(f'ROSS chose {ross_response}.')
        print(f'You chose {hand_gesture}.')

        # Determine the winner
        if hand_gesture == ross_response:
            print("It's a tie!")
        elif (hand_gesture == "rock" and ross_response == "scissors") or \
             (hand_gesture == "paper" and ross_response == "rock") or \
             (hand_gesture == "scissors" and ross_response == "paper"):
            print("You win this round!")
            player_wins += 1
        else:
            print("ROSS wins this round!")
            ross_wins += 1

    # Print the final score
    if player_wins > ross_wins:
        print("You win the game!")
    elif player_wins < ross_wins:
        print("ROSS wins the game!")
    else:
        print("It's a tie game!")

    # Ask if the player wants to play again
    response = input("Do you want to play again? (y/n) ")
    if response.lower() == "y":
        play_game()

# Define the hand gesture recognition function
def recognize_hand_gesture(frame, rock_paper_scissors_dataset):
    cv2.imwrite('image.jpg', frame)
    os.system('mv image.jpg jetson-inference/python/training/classification/; cd jetson-inference/python/training/classification/; imagenet.py --model=models/RPS/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=data/RPS/labels.txt image.jpg image.jpg > file.txt; mv file.txt ~/Desktop/ROSS/RPS/file.txt; video-viewer image.jpg webrtc://@:1234 &')
    f = open('/home/nvidia/Desktop/ROSS/RPS/file.txt', 'r').read()
    os.system('rm /home/nvidia/Desktop/ROSS/RPS/file.txt')
    if 'rock' in f:
        return 'rock'
    elif 'paper' in f:
        return 'paper'
    elif 'scissors' in f:
        return 'scissors'
    else:
        return 'none'

# Start the game
print("Welcome to Rock-Paper-Scissors!")
print("Type 'lets play rock paper scissors' or 'rps' to start the game.")
while True:
    user_input = input("> ")
    if user_input.lower() == "lets play rock paper scissors" or user_input.lower() == "rps":
        play_game()
    else:
        print("Invalid input. Try again!")
```

## Running this project

Running this project
Open VS Code and connect to your Jetson Nano.

Install Jetson Inference and Docker Image:

Follow the instructions from Jetson Inference Installation.
Download the required files:

Drag the downloaded files into your library on the device where you're running the program.
Ensure the files are located in the jetson-inference/python/training/classification/data directory.
Change directory:

Open a terminal and navigate to the nvidia/jetson-inference/ directory:
bash
Copy code
cd nvidia/jetson-inference/
Run the Docker container:

Execute the following command to run the Docker container:
bash
Copy code
./docker/run.sh
You may need to re-enter your NVIDIA password at this step.
Change directory inside the Docker container:

Navigate to the classification directory:
bash
Copy code
cd jetson-inference/python/training/classification
Run the training script:

Re-train the network using the training script:
bash
Copy code
python3 train.py --model-dir=models/RPS data/RPS
Note: This process may take several hours depending on the number of epochs and batch sizes.
Adjust training parameters (optional):

Specify the number of epochs and batch sizes:
bash
Copy code
python3 train.py --model-dir=models/RPS data/RPS --batch-size=NumberOfBatchFiles --workers=NumberOfWorkers --epochs=NumberOfEpochs
Stop and resume training (optional):

You can stop the training at any time using Ctrl+C.
To resume training, use the --resume and --epoch-start flags.
Export the model to ONNX format:

Run the ONNX export script:
bash
Copy code
python3 onnx_export.py --model-dir=models/RPS
Verify the exported model:

Check if the resnet18.onnx model is in the models/RPS directory.
Exit the Docker container:

Press Ctrl + D to exit the Docker container.
Verify the model on your Nano:

Navigate to the jetson-inference/python/training/classification directory on your Nano:
bash
Copy code
cd jetson-inference/python/training/classification
Use the following command to list the files in the models/RPS/ directory and ensure that resnet18.onnx is present:
bash
Copy code
ls models/RPS/
Set environment variables:

Set the NET and DATASET variables:
bash
Copy code
NET=models/RPS
DATASET=data/RPS
Test the model on an image:

Run this command to test the model on an image from the test folder:
bash
Copy code
imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/name/name.jpg TestImage1.jpg
Install the required Python libraries:

Install the necessary libraries for the project:
bash
Copy code
pip install opencv-python numpy
Save the provided code into a file:

Save the following code into a file named rps_game.py.


[View a video explanation here](video link)# ROSS_RPS
