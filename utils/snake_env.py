import pybullet as p
import pybullet_data
import time
import math

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF('plane.urdf')
snake = p.loadURDF('snake.urdf')

anisotropicFriction = [1, 0.01, 0.01]
p.changeDynamics(snake, -1, lateralFriction=2, anisotropicFriction=anisotropicFriction)
numJoints = p.getNumJoints(snake)
for i in range(numJoints):
    p.changeDynamics(snake, i, lateralFriction=2, anisotropicFriction=anisotropicFriction)

width = 224
height = 224

dt = 1./240.
wavePeriod = 0.5
waveLength = 1.5
waveAmplitude = 0.4
waveFront = 0.0
segmentLength = 5.0
forward = True
scaleStart = 1.0

while True:
    # images = p.getCameraImage(width, height)

    if waveFront < segmentLength * 4.0:
        scaleStart = waveFront / segmentLength * 4.0

    # Moving a sin wave down the body of the snake.
    for joint in range(numJoints):
        segment = joint
        phase = (waveFront - (segment + 1) * segmentLength) / waveLength
        phase -= math.floor(phase)
        phase *= math.pi * 2.0

        # Map phase to curvature
        targetPos = math.sin(phase) * scaleStart * waveAmplitude

        p.setJointMotorControl2(snake, joint, p.POSITION_CONTROL,
                                targetPosition=targetPos, force=10)

    # Wave keeps track of where the wave is in time.
    waveFront += dt / wavePeriod * waveLength
    p.stepSimulation()
    time.sleep(dt)
