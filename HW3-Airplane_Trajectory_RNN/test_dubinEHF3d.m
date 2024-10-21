clear
clc
close all


% keep the init pos zero:
x1 = 0; 
y1 = 0;
alt1 = 0;

psi1 = 20*pi/180; % initial heading, between [0, 2*pi]

% change these variables to get different paths
gamma = -30*pi/180; % climb angle, keep in between [-30 deg, 30 deg]
x2 = -0; 
y2 = -500;

% keep these constant
steplenght = 10; % trajectory discretization level
r_min = 100; % vehicle turn radius.


[path, psi_end, num_path_points] = dubinEHF3d(x1, y1, alt1, psi1, x2, y2, r_min, steplenght, gamma);

x = path(1:num_path_points,1);
y = path(1:num_path_points,2);
z = path(1:num_path_points,3);

figure();
plot3(path(1:num_path_points,1),  path(1:num_path_points,2), path(1:num_path_points,3), 'b.-' ); 
hold on; grid on;
plot3(x1,y1,alt1, 'r*')
plot(x2, y2, 'm*')
axis equal
xlabel('x')
ylabel('y')
zlabel('alt')
