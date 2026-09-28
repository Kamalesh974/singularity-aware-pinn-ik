%% Franka Panda (7-DOF) simulation using MathWorks Robotics System Toolbox
% Built from the same modified-DH parameters already verified in the
% Python/PyTorch model (pinn_ik/robot.py). Not run/tested in MATLAB --
% verify step by step before presenting.

clear; clc; close all;

%% Step 1: Create the empty robot tree
robot = rigidBodyTree('DataFormat', 'row', 'MaxNumBodies', 8);

%% Step 2: Modified DH parameters -- [a alpha d theta_offset] per joint
dhparams = [0        0        0.333   0;
            0       -pi/2     0       0;
            0        pi/2     0.316   0;
            0.0825   pi/2     0       0;
           -0.0825  -pi/2     0.384   0;
            0        pi/2     0       0;
            0.088    pi/2     0       0];

%% Step 3: Build the chain, one joint at a time
bodyNames  = {'body1','body2','body3','body4','body5','body6','body7'};
jointNames = {'joint1','joint2','joint3','joint4','joint5','joint6','joint7'};
parentName = 'base';

for i = 1:7
    body = rigidBody(bodyNames{i});
    jnt = rigidBodyJoint(jointNames{i}, 'revolute');
    setFixedTransform(jnt, dhparams(i,:), 'mdh');   % 'mdh' = modified DH
    body.Joint = jnt;
    addBody(robot, body, parentName);
    parentName = bodyNames{i};
end

%% Step 4: Add the fixed flange offset (no joint -- rigid step to the hand point)
flange = rigidBody('flange');
flangeJoint = rigidBodyJoint('flangeJoint', 'fixed');
setFixedTransform(flangeJoint, trvec2tform([0 0 0.107]));
flange.Joint = flangeJoint;
addBody(robot, flange, 'body7');

%% Step 5: Set the joint limits (radians)
limits = [-2.8973  2.8973;
          -1.7628  1.7628;
          -2.8973  2.8973;
          -3.0718 -0.0698;
          -2.8973  2.8973;
          -0.0175  3.7525;
          -2.8973  2.8973];

for i = 1:7
    robot.Bodies{i}.Joint.PositionLimits = limits(i,:);
end

%% Step 6: Test forward kinematics
qTest = [0 0 0 -1.5708 0 1.8675 0];
T = getTransform(robot, qTest, 'flange');
handPos = T(1:3,4)';
fprintf('MATLAB hand position (x, y, z): %.6f, %.6f, %.6f\n', handPos);

% Reference value already computed from the verified Python model, for
% this exact qTest -- should match the line above to ~5 decimal places:
pyRef = [0.581938, 0.000000, 0.654902];
fprintf('Python  hand position (x, y, z): %.6f, %.6f, %.6f\n', pyRef);
fprintf('Difference: %.6f mm\n', norm(handPos - pyRef) * 1000);

%% Step 7: Visualize
figure;
show(robot, qTest);
title('Franka Panda -- qTest configuration');

%% Step 8: Animate a movement (for your demo)
q1 = zeros(1,7);
q2 = [0.3 -0.2 0.1 -2.0 0.2 2.5 0.4];
figure;
for t = linspace(0,1,50)
    qInterp = q1 + t*(q2 - q1);
    show(robot, qInterp, 'PreservePlot', false, 'Frames', 'off');
    drawnow;
end