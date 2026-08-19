function elapsed = stretch_get_energy_v202(matching_kernel_string, lumen_radius_in_microns_range, vessel_wall_thickness_in_microns, microns_per_voxel, pixels_per_sigma_PSF, max_voxels_per_node, data_directory, original_handle, energy_handle, gaussian_to_ideal_ratio, spherical_to_annular_ratio)
%STRETCH_GET_ENERGY_V202 Scratch-only E14 wrapper. Python owns dest/I/O.
% Force MATLAB get_energy_V202 orientations: radii column, microns/PSF 1x3 rows.
lumen_radius_in_microns_range = double(lumen_radius_in_microns_range(:));
microns_per_voxel = double(microns_per_voxel(:)).';
pixels_per_sigma_PSF = double(pixels_per_sigma_PSF(:)).';
vessel_wall_thickness_in_microns = double(vessel_wall_thickness_in_microns);
max_voxels_per_node = double(max_voxels_per_node);
gaussian_to_ideal_ratio = double(gaussian_to_ideal_ratio);
spherical_to_annular_ratio = double(spherical_to_annular_ratio);
data_directory = char(data_directory);
original_handle = char(original_handle);
energy_handle = char(energy_handle);
matching_kernel_string = char(matching_kernel_string);

log_path = fullfile(data_directory, 'e14_matlab.log');
fid = fopen(log_path, 'a');
if fid > 0
    fprintf(fid, 'start %s radii=%d microns=[%g %g %g]\n', datestr(now, 30), ...
        numel(lumen_radius_in_microns_range), microns_per_voxel(1), microns_per_voxel(2), microns_per_voxel(3));
    fclose(fid);
end
tic;
get_energy_V202( ...
    matching_kernel_string, lumen_radius_in_microns_range, ...
    vessel_wall_thickness_in_microns, microns_per_voxel, ...
    pixels_per_sigma_PSF, max_voxels_per_node, data_directory, ...
    original_handle, energy_handle, ...
    gaussian_to_ideal_ratio, spherical_to_annular_ratio);
elapsed = toc;
fid = fopen(log_path, 'a');
if fid > 0
    fprintf(fid, 'done elapsed=%.3f\n', elapsed);
    fclose(fid);
end
end
