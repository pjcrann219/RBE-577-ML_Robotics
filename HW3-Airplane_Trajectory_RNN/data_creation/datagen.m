clear
clc
close all

tic
% keep the init pos zero:
x1 = 0; 
y1 = 0;
alt1 = 0;

psi1 = 20*pi/180; % initial heading, between [0, 2*pi]

% change these variables to get different paths
gamma = -30*pi/180; % climb angle, keep in between [-30 deg, 30 deg]
x2 = -100; 
y2 = -100;

% keep these constant
steplenght = 50; % trajectory discretization level
r_min = 100; % vehicle turn radius.

grid_x = 1000;
grid_y = 1000;
x2 = grid_x *-1;
y2 = grid_y *-1;
grid_square = 20;
delta_gamma = 10 ; 
delta_gamma_rad = delta_gamma * pi/180;
delta_heading = 20 ; %
delta_heading_rad = delta_heading * pi/180;

ncol = (grid_x/grid_square)*(grid_y/grid_square)*(60/delta_gamma)*(360/delta_heading)
[x_grid, y_grid] = meshgrid(x1 + (-grid_x:grid_square:grid_x), y1 + (-grid_y:grid_square:grid_y));
%mrow = (grid_y/grid_square);
data = zeros(ncol, 7);
k = 1;
good_sols = 0
pos_num =1;
start = toc;
fprintf('Time at program start %.6f\n', toc)
for b = 1:(360/delta_heading)
    toc
    heading_delta = psi1 + delta_heading
    for a = 1:(60/delta_gamma)
        gamma_delta = gamma +delta_gamma;
        for i = 1:((grid_y/grid_square)*2)
            y_delta = y2 + grid_square*(i-1);
            for j = 1:((grid_x/grid_square)*2)
                x_delta = x2 + grid_square*(j-1);
                %[path, psi_end, num_path_points] = dubinEHF3d(x1, x2, alt1, heading_delta, x_delta, y_delta, r_min, steplenght, gamma_delta);
                %fprintf('Time Before first function call %.6f\n', toc)
                %funct = toc;
                [path, psi_end, num_path_points] = dubinEHF3d(x1, x2, alt1, heading_delta, x_delta, y_delta, r_min, steplenght, gamma_delta);
                %time_for_func = toc-funct;
                %fprintf('Time function takes %.6f\n', time_for_func)
                if num_path_points ~= 0 
                    data(k:k + num_path_points - 1, 1:3) = path(1:num_path_points, 1:3);
                    data(k:k + num_path_points - 1, 4) = x_delta; % Store x_delta
                    data(k:k + num_path_points - 1, 5) = y_delta; % Store y_delta
                    data(k:k + num_path_points - 1, 6) = heading_delta; % Store heading_delta
                    data(k:k + num_path_points - 1, 7) = gamma_delta; % Store gamma_delta
                    
                    k = k + num_path_points; % Increment k for the next set of stored points
                    percent_done = (k / ncol); %/ 100; % Calculate the percentage done
                    fprintf('%.2f%% Finished\n', percent_done); % Print the percentage
                    good_sols = good_sols +1;
                    %time_for_write = toc-time_for_func;
                   % fprintf('Time function takes %.6f\n', time_for_write)
                else
                    fprintf("\nNo solution Stored\n")
                end
        
                % data(1:3, k) = path(1:num_path_points,1:3);
        
                    %data(1:3, k) = path(1:num_path_points,1:3);
        
            end
        end
    end
end
toc
save('dataArray_v2.mat', 'data');
writematrix(data, 'dataArray_v2.csv');
data_table = array2table(data, 'VariableNames', {'X', 'Y', 'Z', 'XDelta', 'YDelta', 'HeadingDelta', 'GammaDelta'});
writetable(data_table, 'dataArray_table_v2.csv');
figure;
plot3(path(1:num_path_points,1),  path(1:num_path_points,2), path(1:num_path_points,3), 'b.-' ); 
hold on; grid on;
plot3(x1,y1,alt1, 'r*')
plot(x2, y2, 'm*')
axis equal
xlabel('x')
ylabel('y')
zlabel('alt')
