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
                initCamDistance=4, initCamPitch=-60, initCamYaw=0, camMovement=False)


def makeAntDataset(numSamples=10000):
    ant = p.loadURDF('urdf/ant.urdf', [0, 0, 2])
    makeDataset(ant, 'ant', numSamples, motionRange=1,
                initCamDistance=4, initCamPitch=-60, initCamYaw=45, camMovement=False)


def makeSnakeDataset(numSamples=10000):
    snake = p.loadURDF('urdf/snake.urdf', [0, 0, 0.25])
    makeDataset(snake, 'snake', numSamples, motionRange=2,
                initCamDistance=3, initCamPitch=-60, initCamYaw=0, camMovement=False)


def makeManipulatorDataset(numSamples=10000):
    orn = p.getQuaternionFromEuler([0, -1.57, 0])
    # Snake .urdf model attached to the ground acts like a manipulator
    manipulator = p.loadURDF('urdf/snake.urdf', [0, 0, 2.25], orn, useFixedBase=True)
    makeDataset(manipulator, 'manipulator', numSamples, motionRange=2,
                initCamDistance=3, initCamPitch=0, initCamYaw=90, camMovement=False)


def makeDataset(robot, robotName, numSamples, motionRange,
                initCamDistance, initCamPitch, initCamYaw, camMovement):
    """
    General function to make a dataset that can be customized for the each robot.
    """
    path = 'data/' + robotName
    trainPath = path + '/train/'
    testPath = path + '/test/'
    jointsBuffer = []
    imagesBuffer = []

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

        if camMovement:
            camYaw = random.randint(-180, 180)
            camPitch = random.randint(initCamPitch-30, initCamPitch+30)
        else:
            camYaw = initCamYaw
            camPitch = initCamPitch

        robotPos, _ = p.getBasePositionAndOrientation(robot)
        p.resetDebugVisualizerCamera(cameraDistance=initCamDistance,
                                    cameraPitch=camPitch,
                                    cameraYaw=camYaw,
                                    cameraTargetPosition=robotPos)

        img = np.reshape(p.getCameraImage(224, 224)[2], (224, 224, 4))[:, :, 0:3]  # BGR
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(filename=f"img{i}.jpg", img=img)

        imagesBuffer.append(img)
        jointsBuffer.append(np.divide(jointPositions, 2.0))

        time.sleep(0.1)
        p.stepSimulation()
    
    jointsBuffer = np.matrix(jointsBuffer, dtype=np.float32)
    imagesBuffer = np.array(imagesBuffer, dtype=np.float32)

    # Create train and test datasets
    jointsTrain = jointsBuffer[:int(0.8 * numSamples)]
    jointsTest = jointsBuffer[-int(0.2 * numSamples):]
    imagesTrain = imagesBuffer[:int(0.8 * numSamples)]
    imagesTest = imagesBuffer[-int(0.2 * numSamples):]

    # create folders
    if not os.path.exists(trainPath):
        os.makedirs(trainPath)
    if not os.path.exists(testPath):
        os.makedirs(testPath)
    
    # files
    jointsTrainFile = os.path.join(path + '/train/', 'joints.npy')
    jointsTestFile = os.path.join(path + '/test/', 'joints.npy')
    imagesTrainFile = os.path.join(path + '/train/', 'images.npy')
    imagesTestFile = os.path.join(path + '/test/', 'images.npy')

    # create files
    with open(jointsTrainFile, 'wb+') as f:
        np.save(f, jointsTrain)
    with open(jointsTestFile, 'wb+') as f:
        np.save(f, jointsTest)
    with open(imagesTrainFile, 'wb+') as f:
        np.save(f, imagesTrain)
    with open(imagesTestFile, 'wb+') as f:
        np.save(f, imagesTest)

    p.disconnect()


makeManipulatorDataset(5000)
