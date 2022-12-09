import pybullet as p
import pybullet_data
import numpy as np
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


def makeDataset(robot, robotName, numSamples, motionRange, camDistance, camPitch, camYaw):
    """
    General function to make a dataset that can be customized to the each robot.
    """
    path = 'data/' + robotName
    jointsBuffer = []
    imgBuffer = []

    numJoints = p.getNumJoints(robot)

    for i in range(numSamples):
        # Print the progress by every 100 iterations
        if i % 100 == 0:
            print(f"Sample {i}/{numSamples}")

        if robotName == 'snake':
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

        camYaw = random.randint(-180, 180)
        camPitch = random.randint(camPitch-30, camPitch+30)
        robotPos, _ = p.getBasePositionAndOrientation(robot)
        p.resetDebugVisualizerCamera(cameraDistance=camDistance,
                                     cameraPitch=camPitch,
                                     cameraYaw=camYaw,
                                     cameraTargetPosition=robotPos)

        img = np.reshape(p.getCameraImage(224, 224)[2], (224, 224, 4))[:, :, 0:3]  # BGR
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(filename=f"img{i}.jpg", img=img)

        imgBuffer.append(img)
        jointsBuffer.append(np.divide(jointPositions, 2.0))

        time.sleep(0.1)
        p.stepSimulation()
    
    jointsBuffer = np.matrix(jointsBuffer, dtype=np.float32)
    imgBuffer = np.array(imgBuffer, dtype=np.float32)

    # create folders
    if not os.path.exists(path):
        os.makedirs(path)
    
    # files
    jointsFile = os.path.join(path, 'joints.npy')
    imgFile = os.path.join(path, 'images.npy')

    # create files
    with open(jointsFile, 'wb+') as f:
        np.save(f, jointsBuffer)
    with open(imgFile, 'wb+') as f:
        np.save(f, imgBuffer)

    p.disconnect()


makeSnakeDataset(5000)
