from copy import deepcopy
import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time
import os
import random

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
plane = p.loadURDF('plane.urdf')


def makeTurtleDataset(numSamples=10000):
    turtle = p.loadURDF('urdf/turtle.urdf', [0, 0, 0.25])
    makeDataset(turtle, 'turtle', numSamples, motionRange=0.3,
                camDistance=4, camPitch=-45, camYaw=0)


def makeAntDataset(numSamples=10000):
    ant = p.loadURDF('urdf/ant.urdf', [0, 0, 2])
    makeDataset(ant, 'ant', numSamples, motionRange=1,
                camDistance=4, camPitch=-60, camYaw=45)


def makeSnakeDataset(numSamples=10000):
    snake = p.loadURDF('urdf/snake.urdf', [0, 0, 0.25])
    makeDataset(snake, 'snake', numSamples, motionRange=2,
                camDistance=3, camPitch=-89, camYaw=0)


def makeManipulatorDataset(numSamples=10000):
    orn = p.getQuaternionFromEuler([0, -1.57, 0])
    manipulator = p.loadURDF('urdf/snake.urdf', [0, 0, 2.25], orn)
    makeDataset(manipulator, 'manipulator', numSamples, motionRange=2,
                camDistance=3, camPitch=0, camYaw=90)

def makeSnakeNpyDataset(sumSamples=10000):
    snake = p.loadURDF('urdf/snake.urdf', [0, 0, 0.25])
    makeSnakeDatasetNpy(snake, sumSamples, motionRange=2,
                camDistance=3, camPitch=-89, camYaw=0)

def makeSnakeDatasetNpy(robot, numSamples, motionRange, camDistance, camPitch, camYaw):
    robotName = 'snake'
    path = 'data/snake'
    joints_buffer = []
    image_buffer = []

    numJoints = p.getNumJoints(robot)
    i = 0
    for i in range(numSamples):
        print(f"sample {i}")
        # Randomize initial orientation for the first segment for snake
        initShift = np.random.rand(1)
        initAngle = np.random.rand(1) * 6.28
        initAngle *= random.choice((-1, 1))
        p.resetBasePositionAndOrientation(robot,
                                            [initShift, 0, 0.25],
                                            p.getQuaternionFromEuler([initAngle, 1.57, 0]))

        jointPositions = np.random.rand(numJoints) * motionRange
        for joint in range(numJoints):
            jointPositions[joint] *= random.choice((-1, 1))
            p.resetJointState(robot, joint, jointPositions[joint])


        robotPos, _ = p.getBasePositionAndOrientation(robot)
        p.resetDebugVisualizerCamera(cameraDistance=camDistance,
                                     cameraPitch=camPitch,
                                     cameraYaw=camYaw,
                                     cameraTargetPosition=robotPos)

        img = np.reshape(p.getCameraImage(224, 224)[2], (224, 224, 4))[:,:,0:3] #BGR
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        image_buffer.append(img)
        joints_buffer.append(np.divide(jointPositions, 2.0))

        time.sleep(0.1)
        p.stepSimulation()
    
    joints_buffer = np.matrix(joints_buffer, dtype=np.float32)
    image_buffer = np.array(image_buffer, dtype=np.float32)

    # create folders
    if not os.path.exists(path):
        os.makedirs(path)
    
    # files
    joints_file = os.path.join(path, 'joints.npy')
    images_file = os.path.join(path, 'images.npy')

    # create files
    with open(joints_file, 'wb+') as f:
        np.save(f, joints_buffer)
    with open(images_file, 'wb+') as f:
        np.save(f, image_buffer)

    p.disconnect()


def makeDataset(robot, robotName, numSamples, motionRange,
                camDistance, camPitch, camYaw):
    filename = '{}.txt'.format(robotName)
    with open(filename, 'w') as f:
        f.truncate(0)

    numJoints = p.getNumJoints(robot)
    i = 0
    while i < numSamples:
        # Randomize initial orientation for the first segment for snake
        if robotName == 'snake':
            initShift = np.random.rand(1)
            initAngle = np.random.rand(1) * 6.28
            initAngle *= random.choice((-1, 1))
            p.resetBasePositionAndOrientation(robot,
                                              [initShift, 0, 0.25],
                                              p.getQuaternionFromEuler([initAngle, 1.57, 0]))

        jointPositions = np.random.rand(numJoints) * motionRange
        for joint in range(numJoints):
            jointPositions[joint] *= random.choice((-1, 1))
            p.resetJointState(robot, joint, jointPositions[joint])

        decimals = 4
        with open(filename, 'a') as f:
            line = 'id: {} joints: '.format(i)
            for joint in jointPositions:
                line += str(round(joint, decimals)) + ' '
            f.write(line + '\n')

        robotPos, _ = p.getBasePositionAndOrientation(robot)
        p.resetDebugVisualizerCamera(cameraDistance=camDistance,
                                     cameraPitch=camPitch,
                                     cameraYaw=camYaw,
                                     cameraTargetPosition=robotPos)

        img = p.getCameraImage(224, 224)[2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('{}/{}.jpg'.format(robotName, i), gray)
        i += 1
        time.sleep(0.1)
        p.stepSimulation()

    p.disconnect()

makeSnakeNpyDataset(1000)
