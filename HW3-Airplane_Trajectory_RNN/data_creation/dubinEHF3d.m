
function [path, psi_end, num_path_points] = dubinEHF3d(east1, north1, alt1, psi1, east2, north2, r, step, gamma)
%DubinsEHF (END HEADING FREE)
%Finds the minimum length Dubin's curve between a start position and
% heading (east, north, alt, psi) and an end position (east, north). Note that the end heading is free.
% gamma is the flight path angle. step is the lenght between path points. 
% OUTPUT:
%   path: N-by-3 array (east, north, alt)
%   psi_end: scalar, heading at the end point.

% Notes:
% 1- The final altitude is not apriori known and it dependes on the path and gamma.
% 2- The heading (psi) is w.r.t east and positive toward the north (not the standard aeronautice heading!)

MAX_NUM_PATH_POINTS = 1000;
path = zeros(MAX_NUM_PATH_POINTS, 3); %initialization
r_sq = r^2;
while psi1<0
    psi1 = psi1+2*pi;
end 
while psi1>=2*pi
    psi1=psi1-2*pi;
end

% Left and right circles about pl
theta_l = psi1 + pi/2;
eastc_l = east1 + r*cos (theta_l);
northc_l = north1 + r*sin(theta_l);

theta_r = psi1 - pi/2;
eastc_r = east1 + r*cos (theta_r);
northc_r = north1+ r*sin(theta_r);

% Distance from p2 to circle centers
d2c_l_sq = (east2-eastc_l)^2+ (north2-northc_l)^2; 
d2c_r_sq=(east2-eastc_r)^2+(north2-northc_r)^2;
d2c_l = sqrt(d2c_l_sq);
d2c_r = sqrt(d2c_r_sq);
if d2c_l<r || d2c_r<r
    psi_end = 0;
    num_path_points = 0;
    fprintf('no solution: distance of pl and p2 is lower that turn radius \n\n');
    return;
end
% Angle from circle left centers to p2
theta_c_l = atan2(north2-northc_l, east2-eastc_l);
if theta_c_l<0
    theta_c_l = theta_c_l + 2*pi;
end

% Angle from circle right centers to p2
theta_c_r = atan2(north2-northc_r, east2-eastc_r);
if theta_c_r<0
    theta_c_r= theta_c_r + 2*pi;
end

% Length of tangent lines
lt_l_sq = d2c_l_sq - r_sq;
lt_r_sq = d2c_r_sq - r_sq; 
lt_l = sqrt( lt_l_sq);
lt_r = sqrt (lt_r_sq);
% Start angle on cirlces
theta_start_l = theta_r;
theta_start_r = theta_l;
% End angle on circles
% cos_theta_d_1-(x_sq+d2c_1_sq-lt_1_sq) / (2*r*d2c_1);
% if abs (cos_theta_d_1)>1
% end
% path=[];
% return;
% theta_d_1=acos (cos_theta_d_1);
theta_d_l = acos(r/d2c_l);

theta_end_l = theta_c_l-theta_d_l;

while theta_end_l < theta_start_l 
    theta_end_l = theta_end_l + 2*pi;
end
while theta_end_l > theta_start_l+2*pi
    theta_end_l  = theta_end_l -2*pi;
end

theta_d_r = acos(r/d2c_r);
theta_end_r = theta_c_r + theta_d_r;
while theta_end_r >  theta_start_r
    theta_end_r = theta_end_r - 2*pi;
end
while theta_end_r < theta_start_r - 2*pi
    theta_end_r = theta_end_r + 2*pi;
end

% Find left and right total distances
% Sarc_1-wrapTo2Pi (theta_end_1-theta_start_1);
% tarc_r=wrapTo2Pi (theta_end_r-theta_start_r);
arc_l  = abs(theta_end_l-theta_start_l); 
arc_r  = abs(theta_end_r-theta_start_r);
arc_length_l = r*arc_l;
arc_length_r = r*arc_r;

total_length_l = arc_length_l + lt_l;
total_length_r = arc_length_r + lt_r;
% Find path
if total_length_l<total_length_r
    %Arc points
    if arc_length_l > 0.1
        theta_step = step/r;
        num_arc_seg = max (2, ceil(arc_l/theta_step));
        angles = linspace(theta_start_l, theta_end_l, num_arc_seg);
        alt_end = alt1 + arc_length_l*tan(gamma);
        altitude =  linspace(alt1, alt_end, num_arc_seg);
        arc_traj = zeros(num_arc_seg, 3); %initialization
        for i=1:num_arc_seg
            arc_traj (i,:)= [eastc_l+r*cos(angles(i)), northc_l+r*sin(angles(i)), altitude(i)];
        end
    else
        arc_traj = [east1, north1, alt1];
        num_arc_seg = 1;
    end

    %Line points
    if lt_l > 0.1 || arc_length_l< 0.1
        num_line_seg = max(2,ceil(lt_l/step));
        alt_begin =  arc_traj(end, 3);
        alt_end = alt_begin + lt_l*tan(gamma);
        line_traj = zeros(num_line_seg,3); % initialization
        line_traj = [linspace(arc_traj(end, 1), east2, num_line_seg)', linspace(arc_traj(end, 2), north2, num_line_seg)', linspace(alt_begin, alt_end, num_line_seg)'];
    else
        line_traj = [ 0 0 0];
        num_line_seg = 0;
    end

else
    %Arc points:
    if arc_length_r>0.1
        theta_step = step/r;
        num_arc_seg = max (2, ceil (arc_r/theta_step));
        angles = linspace(theta_start_r, theta_end_r, num_arc_seg);
        alt_end = alt1 + arc_length_r*tan(gamma);
        altitude =  linspace(alt1, alt_end, num_arc_seg);
        arc_traj = zeros(num_arc_seg, 3); %initialization
        for i=1:num_arc_seg
            arc_traj (i,:)= [eastc_r+r*cos(angles(i)), northc_r+r*sin(angles(i)), altitude(i)];
        end
    else
        arc_traj = [east1, north1, alt1];
        num_arc_seg = 1;
    end
     %Line points
    if lt_r>0.1 || arc_length_r< 0.1
        num_line_seg = max(2,ceil(lt_r/step));
        alt_begin =  arc_traj (end, 3);
        alt_end = alt_begin + lt_r*tan(gamma);
        line_traj = zeros(num_line_seg,3); % initialization
        line_traj = [linspace(arc_traj(end, 1), east2, num_line_seg)', linspace(arc_traj(end, 2), north2, num_line_seg)', linspace(alt_begin, alt_end, num_line_seg)'];
    else
        line_traj = [ 0 0 0];
        num_line_seg = 0;
    end
        
end

% Final path
if num_line_seg > 1
    num_path_points = num_arc_seg + num_line_seg -1; % minus 1 is because of overlapped point
    path(1:num_path_points,:) =[arc_traj; line_traj(2:end, :)];
else % line does not exist in the path
    num_path_points = num_arc_seg;
    path(1:num_path_points, :)= arc_traj;
end
%the final heading obtained by connecting the last acr point and the goal point
psi_end = atan2(north2-arc_traj(end, 2), east2-arc_traj(end, 1) );

return;

end

