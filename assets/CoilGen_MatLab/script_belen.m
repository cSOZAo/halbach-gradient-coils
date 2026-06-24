%% Script Belen

%% Dfinicion de campo  objetivo

rotx = @(t) [1 0 0; 0 cos(t) -sin(t) ; 0 sin(t) cos(t)] ;
roty = @(t) [cos(t) 0 sin(t) ; 0 1 0 ; -sin(t) 0  cos(t)] ;
rotz = @(t) [cos(t) -sin(t) 0 ; sin(t) cos(t) 0 ; 0 0 1] ;

% rotation for linears
rot_ang_1 = 0;%-atan(sqrt(2));% define magic angle
rot_ang_2 = pi/2;%1*pi/180*120; % define 120° rotation
rot_ang_3 = pi/2;%3*pi/180*90;%-atan(sqrt(2));%; % define 120° rotation

rot1 = rotx(0);                                                                          
rot2 = rotz(pi/2);
rot3 = roty(-pi/2);

vec_x = rot1*([1;0;0]);
vec_y = rot2*([1;0;0]);
vec_z = rot3*([1;0;0]);

% check for orthogonality (should be zero, or just rounding errors)
dot(vec_x,vec_y)
dot(vec_x,vec_z)
dot(vec_y,vec_z)

% Define size of target regions
rx = 0.02;
ry = 0.02;
rz = 0.02;

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


% Definir el directorio 
output_dir = '/Users/seba9/OneDrive/Documentos/MATLAB/Coilgen-main/target_fields';

% Verificar si el directorio existe
if ~exist(output_dir, 'dir')
    error('El directorio de salida no existe: %s', output_dir);
end

% guardamos

straight_grid_out.coords = [target_rot1(1,:);target_rot1(2,:);target_rot1(3,:)];
% straight_grid_out.lin_1 = x3(1,:);
straight_grid_out.lin_1 = x1(1,:);
save(fullfile(output_dir, 'OSI2_GradTarget_Lin1.mat'), 'straight_grid_out');

straight_grid_out.coords = [target_rot2(1,:);target_rot2(2,:);target_rot2(3,:)];
% straight_grid_out.lin_2 = x3(1,:);
straight_grid_out.lin_2 = x1(1,:);
save(fullfile(output_dir, 'OSI2_GradTarget_Lin2.mat'), 'straight_grid_out');


straight_grid_out.coords = [target_rot3(1,:);target_rot3(2,:);target_rot3(3,:)];
% straight_grid_out.lin_3 = x3(1,:);
straight_grid_out.lin_3 = x1(1,:);
save(fullfile(output_dir, 'OSI2_GradTarget_Lin3.mat'), 'straight_grid_out');


% vemos las direcciones
figure;
hold all
plot3([0 vec_x(1)],[0 vec_x(2)], [0 vec_x(3)],'g')
plot3([0 vec_y(1)],[0 vec_y(2)], [0 vec_y(3)],'g')
plot3([0 vec_z(1)],[0 vec_z(2)], [0 vec_z(3)],'g')



xlim([-1 1])
ylim([-1 1])
zlim([-1 1])

axis tight equal 
xlabel('x')
ylabel('y')
zlabel('z')

%% Visualizamos los campos objetivo
% cargamos el target field
load('/Users/seba9/OneDrive/Documentos/MATLAB/Coilgen-main/target_fields/OSI2_GradTarget_Lin1.mat');


% coordenadas
X = straight_grid_out.coords(1,:)';
Y = straight_grid_out.coords(2,:)';
Z = straight_grid_out.coords(3,:)';

% Extraer componente del campo objetivo 
Bmag = straight_grid_out.lin_1';  


U = Bmag;
V = zeros(size(U));
W = zeros(size(U));

% Visualizar con quiver3
figure;
quiver3(X, Y, Z, U, V, W, 'AutoScale', 'on', 'AutoScaleFactor', 5, 'Color', 'b');
axis equal;
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Gradiente objetivo: lin\_1 (sin rotación)');
grid on;



% cargamos el target field
load('/Users/seba9/OneDrive/Documentos/MATLAB/Coilgen-main/target_fields/OSI2_GradTarget_Lin2.mat');


X = straight_grid_out.coords(1,:)';
Y = straight_grid_out.coords(2,:)';
Z = straight_grid_out.coords(3,:)';

Bmag = straight_grid_out.lin_2';  


U = Bmag;
V = zeros(size(U));
W = zeros(size(U));

% Visualizar con quiver3
figure;
quiver3(X, Y, Z, U, V, W, 'AutoScale', 'on', 'AutoScaleFactor', 5, 'Color', 'b');
axis equal;
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Gradiente objetivo: lin\_2 (sin rotación)');
grid on;


load('/Users/seba9/OneDrive/Documentos/MATLAB/Coilgen-main/target_fields/OSI2_GradTarget_Lin3.mat');


X = straight_grid_out.coords(1,:)';
Y = straight_grid_out.coords(2,:)';
Z = straight_grid_out.coords(3,:)';


Bmag = straight_grid_out.lin_3'; 
% Si estás usando dirección fija (por ejemplo x), define así:
U = Bmag;
V = zeros(size(U));
W = zeros(size(U));

% Visualizar con quiver3
figure;
quiver3(X, Y, Z, U, V, W, 'AutoScale', 'on', 'AutoScaleFactor', 5, 'Color', 'b');
axis equal;
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Gradiente objetivo: lin\_3 (sin rotación)');
grid on;

%% seccion transversal del cable 

% para generar un zurco y que el cale quede bien, vamos a usar una seccion
% ovalada 

% Parámetros
circular_resolution = 11;
conductor_width = 0.00112; %Diametro de alambre de cobre 1.10mm, Alambre cable Litz 1.55 - 1.90mm
largo_ovalo = 1; %Valor que controla el largo del óvalo, cuando es 0 es una esfera
% --- Sección ovalada: esta es la FINAL (cross_sectional_points)
cross_sectional_points = [ ...
    0.7*sin(0:(2*pi)/(circular_resolution-1):pi), -largo_ovalo + 0.7*sin(pi:(2*pi)/(circular_resolution-1):2*pi);
    0.7*cos(0:(2*pi)/(circular_resolution-1):pi),       0.7*cos(pi:(2*pi)/(circular_resolution-1):2*pi)];
cross_sectional_points = [cross_sectional_points cross_sectional_points(:,1)]; % cerrar curva
cross_sectional_points = cross_sectional_points * conductor_width;

% --- Sección "real" anterior para comparar
R_section = conductor_width / 1.55; % Radio = 0.00075 m
theta = linspace(0, 2*pi, 100); % Usa 100 puntos para un círculo más suave

x_coords_new = R_section * cos(theta);
y_coords_new = R_section * sin(theta);
manual_section = [x_coords_new; y_coords_new];

% --- Graficar ambas
figure;
hold on;
plot(cross_sectional_points(1,:), cross_sectional_points(2,:), 'r-', 'LineWidth', 2);
plot(manual_section(1,:), manual_section(2,:), 'b--', 'LineWidth', 2);
legend('Sección ovalada (cross\_sectional\_points)', 'Sección manual anterior');
axis equal;
xlabel('x (m)');
ylabel('y (m)');
title('Comparación de secciones transversales del conductor');
grid on;

%% Parametros

cut_width=0.001;
min_loop_signifcance=3 ; % Mínimo valor (en unidades de función stream)
%  para considerar una línea de corriente como "válida" para construir una vuelta.

pot_offset_factor = 0.9;
%controla qué tan cerca del máximo/mínimo de la función de corriente se colocan las vueltas.
%Valores altos (como 0.9) evitan los extremos, haciendo el diseño más estable pero con menor cobertura del campo máximo.
%Es útil para evitar cruces o bobinas inestables en los bordes.

target_field_definition_file='/Users/seba9/OneDrive/Documentos/MATLAB/Coilgen-main/target_fields/OSI2_GradTarget_Lin2.mat';

normal_shift_smooth_factors=[7 7 7];
%Valores mayores → más suavizado → menor riesgo de cruce.
%Pero si es demasiado alto, puede distorsionar el layout.
% este es el que recomiendan en el paper

conductor_width=0.00112; % nuestro cable 1.50 cobre, 2.31 litz

temp.init=[]; %this struct must be set for the first time

radii = [0.045, 0.0515, 0.058]; % el primero, radii[0] es el de prueba

num_levels = 15; 
%Define cuántos contornos de corriente (vueltas) tendrá la bobina.
% Menos niveles → menos vueltas → diseño más simple, pero el campo puede ser menos uniforme o preciso.
%Más niveles → más vueltas → mejor fidelidad del campo, pero mas complejidad y riesgo de cruces.

normal_shift=-0.003;
%normal_shift separa físicamente los caminos de ida y retorno para que no se crucen ni se solapen

tikonov_factor= 3*10^3; 
%Controla la suavidad de la función de corriente optimizada mediante regularización de Tikhonov.
% Valor alto (como 3000) → suaviza la solución → pistas más limpias, menos deEtalladas.
%Valor bajo → sigue más fielmente el campo objetivo, pero puede generar layouts irregulares o con caminos entrecruzados

num_circular_divisions = 100; %Modifica la resolución que tendrá la bobina al añadir una a mayor cantidad de secciones transversales 
%Esto altera el tiempo que tarda en generarse la bobina considerablementecomo también los resultados del gradiente


    %% Ejecutamos



% Define el archivo de definición de campo objetivo y su directorio
target_field_definition_file = 'OSI2_GradTarget_Lin3.mat';
target_field_dir = '/Users/seba9/OneDrive/Documentos/MATLAB/Coilgen-main/target_fields';

coil_1.out = CoilGen(... 
    'temp', temp, ...
    'target_field_definition_file', target_field_definition_file, ...
    'target_field_definition_field_name', 'lin_3', ...
    'coil_mesh_file', 'create cylinder mesh', ...    
    'cylinder_mesh_parameter_list', [0.1 radii(2) num_circular_divisions 10 0 1 0 pi/2], ... % cylinder_height[in m], cylinder_radius[in m], num_circular_divisions,  num_longitudinal_divisions, rotation_vector: x,y,z, and  rotation_angle [radian]
    'surface_is_cylinder_flag', true, ...
    'min_loop_signifcance', min_loop_signifcance, ...
    'levels', num_levels, ... % the number of potential steps that determines the later number of windings (Stream function discretization)
    'pot_offset_factor', pot_offset_factor, ... % a potential offset value for the minimal and maximal contour potential ; must be between 0 and 1
    'interconnection_cut_width', cut_width, ... % the width for the interconnections are interconnected; in meter
    'conductor_cross_section_width', conductor_width, ... %width of the generated pcb tracks
    'normal_shift_length', normal_shift, ... % the length for which overlapping return paths will be shifted along the surface normals; in meter
    'skip_postprocessing', false, ...
    'field_shape','z',...
    'make_cylndrical_pcb', false, ...
    'skip_inductance_calculation', true, ...
    'cross_sectional_points', cross_sectional_points, ...    
    'normal_shift_smooth_factors', normal_shift_smooth_factors, ...
    'smooth_flag', true, ...
    'smooth_factor', 3, ...
    'save_stl_flag', true, ...
    'tikonov_reg_factor', tikonov_factor); %Tikonov regularization factor for the SF optimization

% Ruta de origen del archivo STL generado por CoilGen
source_stl_file = '/Users/seba9/OneDrive/Documentos/MATLAB/Coilgen-main/sweeped_layout_part1_z.stl';
% Ruta de destino del archivo STL
destination_stl_file = fullfile(target_field_dir, ['Belen_Y_v2', num2str(tikonov_factor), '_Ch_y_', num2str(num_levels), '_res92_42_90c.stl']);

% Mover el archivo STL a la ubicación de destino
if exist(source_stl_file, 'file') == 2
    movefile(source_stl_file, destination_stl_file);
else
    error('El archivo STL no se encontró en la ruta esperada: %s', source_stl_file);
end


%% Plot results
coil_name='A4IM_LIN';

if ispc
addpath(strcat(pwd,'\','plotting'));
else
addpath(strcat(pwd,'/','plotting'));
end

for coil_to_plot = [coil_1];

 single_ind_to_plot= find_even_leveled_solution(coil_to_plot);

 plot_error_different_solutions(coil_to_plot,single_ind_to_plot,coil_name);
 plot_2D_contours_with_sf(coil_to_plot,single_ind_to_plot,coil_name);
 plot_3D_sf(coil_to_plot,single_ind_to_plot,coil_name);
 plot_groups_and_interconnections(coil_to_plot,single_ind_to_plot,coil_name);

 plot_coil_parameters(coil_to_plot,coil_name);
 plot_coil_track_with_resulting_bfield(coil_to_plot,single_ind_to_plot,coil_name);
 plot_various_error_metrics(coil_to_plot,single_ind_to_plot,coil_name);
 plot_resulting_gradient(coil_to_plot,single_ind_to_plot,coil_name);


 

end

%% 
G = coil_to_plot(single_ind_to_plot).out.layout_gradient.gradient_in_target_direction;
meanG = mean(G);
stdG  = std(G);
cvG   = stdG / abs(meanG);   % coeficiente de variación (std/mean)

% fracciones dentro de tolerancias relativas (ej. 1%, 2%, 5%)
tol_list = [0.01, 0.02, 0.05];
for k = 1:numel(tol_list)
    tol = tol_list(k);
    frac = sum(abs(G - meanG) <= tol*abs(meanG)) / numel(G);
    fprintf('Frac within ±%.2f%% = %.3f\n', tol*100, frac);
end

% valores extremos
minG = min(G);
maxG = max(G);
fprintf('mean=%.4f mT/m/A, std=%.4f, CV=%.4f, min=%.4f, max=%.4f\n', meanG, stdG, cvG, minG, maxG);

%% VISUALIZACIÓN DE GEOMETRÍA (MÉTODO ROBUSTO)

if isfield(coil_1.out, 'coil_parts') && ~isempty(coil_1.out.coil_parts)
    
    % 1. Dato oficial (Ya lo tienes, pero lo mostramos por completitud)
    largo_oficial = coil_1.out.coil_parts(1).coil_length;
    fprintf('\n>>> LARGO OFICIAL (COILGEN): %.4f metros <<<\n', largo_oficial);
    
    % 2. Extracción de coordenadas para gráfico
    raw_data = coil_1.out.coil_parts(1).wire_path;
    puntos_finales = [];
    
    % --- Lógica de "Taladro" para encontrar los números ---
    temp_data = raw_data;
    max_intentos = 5; % Evitar bucles infinitos
    contador = 0;
    
    % Mientras sea una celda, seguimos entrando
    while iscell(temp_data) && contador < max_intentos
        if isempty(temp_data)
            warning('La celda wire_path está vacía en el nivel %d.', contador);
            break;
        end
        % Si hay más de un segmento (celda de celdas), tomamos el primero o concatenamos
        % Generalmente wire_path es un solo camino continuo.
        temp_data = temp_data{1}; 
        contador = contador + 1;
    end
    
    % Verificamos si logramos llegar a los números
    if isnumeric(temp_data)
        puntos_finales = temp_data;
        fprintf('Éxito: Se encontraron datos numéricos tras descender %d niveles.\n', contador);
    else
        warning('No se pudo extraer una matriz numérica. Tipo final: %s', class(temp_data));
    end
    
    % --- Gráfico y Validación Manual ---
    if ~isempty(puntos_finales)
        % Orientación [N x 3]
        [r, c] = size(puntos_finales);
        if r == 3 && c > 3
            puntos_finales = puntos_finales';
        end
        
        % Cálculo simple
        diferencias = diff(puntos_finales, 1, 1);
        distancias = sqrt(sum(diferencias.^2, 2));
        largo_grafico = sum(distancias);
        
        fprintf('Largo calculado del gráfico: %.4f metros\n', largo_grafico);
        
        % Plot
        figure('Name', 'Validación de Geometría');
        plot3(puntos_finales(:,1), puntos_finales(:,2), puntos_finales(:,3), 'b.-');
        grid on; axis equal;
        title(['Geometría del Bobinado (Largo: ' num2str(largo_oficial) ' m)']);
        xlabel('X'); ylabel('Y'); zlabel('Z');
        view(3);
    end
else
    error('Estructura vacía.');
end