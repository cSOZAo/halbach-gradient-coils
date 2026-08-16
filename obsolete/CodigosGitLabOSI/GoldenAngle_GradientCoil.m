% This script generates a golden angle gradient coil set for a halbach
% magnet. Orientation of the magnet is along y- (up-down) direction.
% It is intended for the OSI system within the A4IM project.
% Autors: Sebastian Littin with the help of Feng Jia and Philipp Amrein, 
% Medical Physics, Department of Diagnostiv and Interventional Radiology, 
% University Medical Center Freiburg, Germany
% July 2024

clear all
clc

addpath(genpath('/home/littin/Documents/git/CoilGen'))
cd('/home/littin/Documents/git/CoilGen')

%% Define target fields rotate by golden angle

rotx = @(t) [1 0 0; 0 cos(t) -sin(t) ; 0 sin(t) cos(t)] ;
roty = @(t) [cos(t) 0 sin(t) ; 0 1 0 ; -sin(t) 0  cos(t)] ;
rotz = @(t) [cos(t) -sin(t) 0 ; sin(t) cos(t) 0 ; 0 0 1] ;

% magic angle rotation
rot_ang_1 = -atan(sqrt(2));% define magic angle
rot_ang_2 = 1*pi/180*120; % define 120° rotation
rot_ang_3 = 3*pi/180*90;%-atan(sqrt(2));%; % define 120° rotation

rot1 = rotx(rot_ang_2*0+rot_ang_3)*roty(rot_ang_1);
rot2 = rotx(rot_ang_2*(-1)+rot_ang_3)*roty(rot_ang_1);
rot3 = rotx(rot_ang_2*1+rot_ang_3)*roty(rot_ang_1);

% % rotation for linears
% rot_ang_1 = 0;%-atan(sqrt(2));% define magic angle
% rot_ang_2 = pi/2;%1*pi/180*120; % define 120° rotation
% rot_ang_3 = pi/2;%3*pi/180*90;%-atan(sqrt(2));%; % define 120° rotation
% 
% rot1 = rotx(0);
% rot2 = rotz(pi/2);
% rot3 = roty(-pi/2);

vec_x = rot1*([1;0;0])
vec_y = rot2*([1;0;0])
vec_z = rot3*([1;0;0])

% check for orthogonality (should be zero, or just rounding errors)
dot(vec_x,vec_y)
dot(vec_x,vec_z)
dot(vec_y,vec_z)

% Define size of target regions
rx = 0.125;
ry = 0.125;
rz = 0.125;

resol_radial = 4;
resol_angular = 28;

% r = [rx,ry,rz];
d1 = 1/(resol_radial-1);
d2 = pi/(resol_angular-1);
d3 = 2*pi/(resol_angular-1);
[ra,theta,phi] = ndgrid(0:d1:1, 0:d2:pi, -pi:d3:pi);

x1 = rx.*ra(:)'.*sin(theta(:)').*cos(phi(:)');
x2 = ry.*ra(:)'.*sin(theta(:)').*sin(phi(:)');
x3 = rz.*ra(:)'.*cos(theta(:)');

r = ra(:)';
phi = theta(:)';
z_rot = phi(:)';

Points = [x1;x2;x3];
target_rot1 = rot1*Points;
target_rot2 = rot2*Points;
target_rot3 = rot3*Points;

straight_grid_out.coords = [target_rot1(1,:);target_rot1(2,:);target_rot1(3,:)];
% straight_grid_out.lin_1 = x3(1,:);
straight_grid_out.lin_1 = x1(1,:);
save '../CoilGen/target_fields/OSI2_GradTarget_GA1.mat' straight_grid_out

straight_grid_out.coords = [target_rot2(1,:);target_rot2(2,:);target_rot2(3,:)];
% straight_grid_out.lin_2 = x3(1,:);
straight_grid_out.lin_2 = x1(1,:);
save '../CoilGen/target_fields/OSI2_GradTarget_GA2.mat' straight_grid_out

straight_grid_out.coords = [target_rot3(1,:);target_rot3(2,:);target_rot3(3,:)];
% straight_grid_out.lin_3 = x3(1,:);
straight_grid_out.lin_3 = x1(1,:);
save '../CoilGen/target_fields/OSI2_GradTarget_GA3.mat' straight_grid_out



figure;
hold all
plot3([0 vec_x(1)],[0 vec_x(2)], [0 vec_x(3)],'g')
plot3([0 vec_y(1)],[0 vec_y(2)], [0 vec_y(3)],'g')
plot3([0 vec_z(1)],[0 vec_z(2)], [0 vec_z(3)],'g')

plot3([0 1],[0 0], [0 0],'r')
plot3([0 0],[0 1], [0 0],'k')
plot3([0 0],[0 0], [0 1],'b')

xlim([-1 1])
ylim([-1 1])
zlim([-1 1])

axis tight equal 
xlabel('x')
ylabel('y')
zlabel('z')



%%

cut_width=0.04;
min_loop_signifcance=2;
pot_offset_factor = 0.5;

normal_shift_smooth_factors=[7 7 7];
circular_resolution=11;
conductor_width=0.002/2;

temp.init=[]; %this struct must be set for the first time

radii = [0.156; 0.1575; 0.159];
% radii = [0.152; 0.155; 0.158];
%% Run the algorithm

% Channel 1

cross_sectional_points = [sin(0:(2*pi)/(circular_resolution-1):pi),-8+sin(pi:(2*pi)/(circular_resolution-1):2*pi); cos(0:(2*pi)/(circular_resolution-1):pi),cos(pi:(2*pi)/(circular_resolution-1):2*pi)];
cross_sectional_points = [cross_sectional_points cross_sectional_points(:,1)];
cross_sectional_points=cross_sectional_points.*repmat(conductor_width,[2 1]);
normal_shift=-0.012;

target_field_definition_file='OSI2_GradTarget_GA1.mat';

num_levels = 42;    
tikonov_factor=3000;

coil_1.out=CoilGen(...
    'temp',temp,...
    'target_field_definition_file',target_field_definition_file,...
    'target_field_definition_field_name','lin_1',...
    'coil_mesh_file','create cylinder mesh', ...    
    'cylinder_mesh_parameter_list',[0.418 radii(1) 50 50 0 1 0 pi/2],... % cylinder_height[in m], cylinder_radius[in m], num_circular_divisions,  num_longitudinal_divisions, rotation_vector: x,y,z, and  rotation_angle [radian]
    'surface_is_cylinder_flag',true, ...
    'min_loop_signifcance',min_loop_signifcance,...
    'levels',num_levels, ... % the number of potential steps that determines the later number of windings (Stream function discretization)
    'pot_offset_factor',pot_offset_factor, ... % a potential offset value for the minimal and maximal contour potential ; must be between 0 and 1
    'interconnection_cut_width',cut_width, ... % the width for the interconnections are interconnected; in meter
    'conductor_cross_section_width',conductor_width,... %width of the generated pcb tracks
    'normal_shift_length',normal_shift, ... % the length for which overlapping return paths will be shifted along the surface normals; in meter
    'skip_postprocessing',false,...
    'make_cylndrical_pcb',false,...
    'skip_inductance_calculation',true,...
    'cross_sectional_points',cross_sectional_points,...    
    'normal_shift_smooth_factors',normal_shift_smooth_factors,...
    'smooth_flag',true,...
    'smooth_factor',1,...
    'save_stl_flag',true,...
    'tikonov_reg_factor',tikonov_factor); %Tikonov regularization factor for the SF optimization

    movefile('../CoilGen/sweeped_layout_part1_x.stl', append('../CoilGen/AIM_GA',num2str(tikonov_factor),'_Ch1_',num2str(num_levels),'_res_92_42.stl'));


%%

cross_sectional_points = [sin(0:(2*pi)/(circular_resolution-1):pi),-5+sin(pi:(2*pi)/(circular_resolution-1):2*pi); cos(0:(2*pi)/(circular_resolution-1):pi),cos(pi:(2*pi)/(circular_resolution-1):2*pi)];
cross_sectional_points = [cross_sectional_points cross_sectional_points(:,1)];
cross_sectional_points=cross_sectional_points.*repmat(conductor_width,[2 1]);
normal_shift=-0.012;

target_field_definition_file='OSI2_GradTarget_GA2.mat';

min_loop_signifcance=6;
num_levels = 42;
    
tikonov_factor=3000;

coil_2.out=CoilGen(...
    'temp',temp,...
    'target_field_definition_file',target_field_definition_file,...
    'target_field_definition_field_name','lin_1',...
    'coil_mesh_file','create cylinder mesh', ...    
    'cylinder_mesh_parameter_list',[0.418 radii(2) 92 42 0 1 0 pi/2],... % cylinder_height[in m], cylinder_radius[in m], num_circular_divisions,  num_longitudinal_divisions, rotation_vector: x,y,z, and  rotation_angle [radian]
    'surface_is_cylinder_flag',true, ...
    'min_loop_signifcance',min_loop_signifcance,...
    'levels',num_levels, ... % the number of potential steps that determines the later number of windings (Stream function discretization)
    'pot_offset_factor',pot_offset_factor, ... % a potential offset value for the minimal and maximal contour potential ; must be between 0 and 1
    'interconnection_cut_width',cut_width, ... % the width for the interconnections are interconnected; in meter
    'conductor_cross_section_width',conductor_width,... %width of the generated pcb tracks
    'normal_shift_length',normal_shift, ... % the length for which overlapping return paths will be shifted along the surface normals; in meter
    'force_cut_selection', {'low' 'low' 'high' 'high' 'high' 'high' 'high' 'high' 'high'},...
    'skip_postprocessing',false,...
    'make_cylndrical_pcb',false,...
    'skip_inductance_calculation',true,...
    'cross_sectional_points',cross_sectional_points,...    
    'normal_shift_smooth_factors',normal_shift_smooth_factors,...
    'smooth_flag',true,...
    'smooth_factor',1,...
    'save_stl_flag',true,...
    'tikonov_reg_factor',tikonov_factor); %Tikonov regularization factor for the SF optimization

movefile('../CoilGen/sweeped_layout_part1_x.stl', append('../CoilGen/AIM_GA',num2str(tikonov_factor),'_Ch2_',num2str(num_levels),'_res_92_42.stl'));


%%
cross_sectional_points = [sin(0:(2*pi)/(circular_resolution-1):pi),-2+sin(pi:(2*pi)/(circular_resolution-1):2*pi); cos(0:(2*pi)/(circular_resolution-1):pi),cos(pi:(2*pi)/(circular_resolution-1):2*pi)];
cross_sectional_points = [cross_sectional_points cross_sectional_points(:,1)];
cross_sectional_points=cross_sectional_points.*repmat(conductor_width,[2 1]);
normal_shift=-0.012;

target_field_definition_file='OSI2_GradTarget_GA3.mat';

min_loop_signifcance=2;
num_levels = 42;
tikonov_factor=3000;

coil_3.out=CoilGen(...
    'temp',temp,...
    'target_field_definition_file',target_field_definition_file,...
    'target_field_definition_field_name','lin_1',...
    'coil_mesh_file','create cylinder mesh', ...    
    'cylinder_mesh_parameter_list',[0.418 radii(3) 92 42 0 1 0 pi/2],... % cylinder_height[in m], cylinder_radius[in m], num_circular_divisions,  num_longitudinal_divisions, rotation_vector: x,y,z, and  rotation_angle [radian]
    'surface_is_cylinder_flag',true, ...
    'min_loop_signifcance',min_loop_signifcance,...
    'levels',num_levels, ... % the number of potential steps that determines the later number of windings (Stream function discretization)
    'pot_offset_factor',pot_offset_factor, ... % a potential offset value for the minimal and maximal contour potential ; must be between 0 and 1
    'interconnection_cut_width',cut_width, ... % the width for the interconnections are interconnected; in meter
    'conductor_cross_section_width',conductor_width,... %width of the generated pcb tracks
    'normal_shift_length',normal_shift, ... % the length for which overlapping return paths will be shifted along the surface normals; in meter
    'skip_postprocessing',false,...
    'make_cylndrical_pcb',false,...
    'skip_inductance_calculation',true,...
    'cross_sectional_points',cross_sectional_points,...    
    'normal_shift_smooth_factors',normal_shift_smooth_factors,...
    'smooth_flag',true,...
    'smooth_factor',1,...
    'save_stl_flag',true,...
    'tikonov_reg_factor',tikonov_factor); %Tikonov regularization factor for the SF optimization

movefile('../CoilGen/sweeped_layout_part1_x.stl', append('../CoilGen/AIM_GA',num2str(tikonov_factor),'_Ch3_',num2str(num_levels),'_res_92_42.stl'));

return
%% Plot results


coil_name='A4IM_GoldenAngleGradient';

for coil_to_plot = [coil_1,coil_2,coil_3];

single_ind_to_plot= find_even_leveled_solution(coil_to_plot);

% plot_error_different_solutions(coils_to_plot,single_ind_to_plot,coil_name);
% plot_2D_contours_with_sf(coils_to_plot,single_ind_to_plot,coil_name);
plot_3D_sf(coil_to_plot,single_ind_to_plot,coil_name);
plot_groups_and_interconnections(coil_to_plot,single_ind_to_plot,coil_name);

% plot_coil_parameters(coils_to_plot,coil_name);
plot_coil_track_with_resulting_bfield(coil_to_plot,single_ind_to_plot,coil_name);
plot_various_error_metrics(coil_to_plot,single_ind_to_plot,coil_name);
plot_resulting_gradient(coil_to_plot,single_ind_to_plot,coil_name);

end
return

