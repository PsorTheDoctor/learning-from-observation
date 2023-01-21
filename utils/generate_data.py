import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time
import random
import csv
import os
import shutil
from utils.const import *

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

plane = p.loadURDF('plane.urdf')
table = p.loadURDF('table/table.urdf')
robotId = p.loadSDF('kuka_iiwa/kuka_with_gripper2.sdf')
robot = robotId[0]
numJoints = p.getNumJoints(robot)
p.resetBasePositionAndOrientation(robot, [0, 0, 0.7],
                                         [0, 0, 0, 1])

robotPos = p.getBasePositionAndOrientation(robot)[0]
p.resetDebugVisualizerCamera(cameraDistance=1.6, cameraYaw=180, cameraPitch=-41,
                             cameraTargetPosition=robotPos)


def randomColor():
    color = np.random.rand(4)
    color[3] = 1
    return color


def createDataset(numSamples, data_source=DATA_TRAIN):

    try:
        shutil.rmtree(data_source)
    except Exception:
        print("file does not exists!")
    os.makedirs(f"{data_source}/images")
    decimals = 4
    filename = f'{data_source}/joints.csv'

    with open(filename, 'w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(numSamples):

            p.changeVisualShape(plane, -1, rgbaColor=randomColor())
            p.changeVisualShape(table, -1, rgbaColor=randomColor())

            color = randomColor()
            jointPositions = np.random.rand(numJoints) * 2.

            for joint in range(numJoints):
                p.changeVisualShape(robot, joint, rgbaColor=color)

                jointPositions[joint] *= random.choice((-1, 1))
                p.resetJointState(robot, joint, jointPositions[joint])

            csv_writer.writerow(np.around(jointPositions, decimals))

            w, h, rgba, depth, mask = p.getCameraImage(224, 224)
            gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
            #normalize
            gray = (np.expand_dims(gray, axis=2)/255).astype(np.float64)
            depth = (np.expand_dims(depth, axis=2)).astype(np.float64)
            # mask = (np.expand_dims(mask, axis=2)/255).astype(np.float64)
            depth_img = np.concatenate((gray, depth), axis=2)
            depth_img.dump(f"{data_source}/images/{i}")

            p.stepSimulation()
    p.disconnect()


createDataset(100, data_source=DATA_TEST)
