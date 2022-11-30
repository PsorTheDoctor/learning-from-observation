import pybullet as p
import pybullet_data
import numpy as np
import cv2
import imutils
from imutils.video import FileVideoStream
import random
import threading
import time
from models.vit import DeepViT


def inputSim(sharedImg):
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF('plane.urdf')
    robot = p.loadURDF('urdf/snake.urdf')

    numJoints = p.getNumJoints(robot)
    motionRange = 2
    jointPositions = np.random.rand(numJoints) * motionRange
    for joint in range(numJoints):
        jointPositions[joint] *= random.choice((-1, 1))
        p.resetJointState(robot, joint, jointPositions[joint])

    width = len(sharedImg)
    height = len(sharedImg[0])
    channels = len(sharedImg[1])

    img = p.getCameraImage(width, height)[2]
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for x in range(width):
        for y in range(height):
            for c in range(channels):
                sharedImg[x][y][c] = img[x][y][c]

    while True:
        p.stepSimulation()
        time.sleep(1./240.)


def inputVideo(filename, sharedImg):
    vs = FileVideoStream(filename)
    while True:
        frame = vs.read()
        imutils.resize(frame, width=len(sharedImg), height=len(sharedImg[0]))
        height, width, channels = frame.shape

        for x in range(width):
            for y in range(height):
                for c in range(channels):
                    sharedImg[x][y][c] = frame[x][y][c]

        cv2.imshow(frame)
        if cv2.waitKey == 27:
            break

    vs.stop()
    cv2.destroyAllWindows()


def estimatePose(img, joints):
    img = np.reshape(img , (1, 256, 256, 3))
    net = DeepViT(
        image_size=256,
        patch_size=32,
        num_classes=2,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    preds = net(img)
    print(preds)
    # preds = np.zeros(2, dtype=np.float32)
    for i in range(len(preds)):
        joints[i] = preds[0][i]


def outputSim(joints):
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF('plane.urdf')
    robot = p.loadURDF('urdf/snake.urdf')

    numJoints = len(joints)
    for i in range(numJoints):
        p.resetJointState(robot, i, joints[i])

    while True:
        p.stepSimulation()
        time.sleep(1./240.)


def img2sim(filename):
    """
    Takes an image as input and set a robot pose basing on it.
    """
    img = cv2.imread(filename)
    img = imutils.resize(img, width=256, height=256)
    joints = [0 for _ in range(2)]
    estimatePose(img, joints)
    outputSim(joints)


def sim2sim():
    """
    Simulation-to-simulation translation.
    """
    # Creating an empty matrix of shape 256x256x3 to store the data from a camera
    sharedImg = [[[0 for _ in range(256)] for _ in range(256)] for _ in range(3)]
    # Defining the number of joints as 2 for snake
    joints = [0 for _ in range(2)]

    inputSimThread = threading.Thread(target=inputSim, data=(sharedImg,))
    visionThread = threading.Thread(target=estimatePose, data=(sharedImg, joints,))
    outputSimThread = threading.Thread(target=outputSim, data=(joints,))

    inputSimThread.start()
    visionThread.start()
    outputSimThread.start()


def video2sim(filename):
    """
    Video-controlled simulation.
    """
    sharedImg = [[[0 for _ in range(256)] for _ in range(256)] for _ in range(3)]
    joints = [0 for _ in range(2)]

    videoThread = threading.Thread(target=inputVideo, data=(filename, sharedImg,))
    visionThread = threading.Thread(target=estimatePose, data=(sharedImg, joints,))
    simThread = threading.Thread(target=outputSim, data=(joints,))

    videoThread.start()
    visionThread.start()
    simThread.start()


if __name__ == '__main__':
    img2sim('images/snake.png')
