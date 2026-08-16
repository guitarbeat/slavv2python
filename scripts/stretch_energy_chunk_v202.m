function [energy, scale_idx] = stretch_energy_chunk_v202(chunk, matching_kernel_string, radii, vessel_wall_thickness_in_microns, microns_per_pixel, pixels_per_sigma_PSF, y0, y1, x0, x1, z0, z1, y_offset, x_offset, z_offset, y_write_count, x_write_count, z_write_count, rf_y, rf_x, rf_z, gaussian_to_ideal_ratio, spherical_to_annular_ratio, scales_per_octave)
%STRETCH_ENERGY_CHUNK_V202 MATLAB-engine per-chunk Energy float body.
% Python still owns chunk lattice, resume, and checkpoint packaging.
% MATLAB owns fourier_transform_V2, energy_filter_V200, interp3 upsample,
% and min-projection across scales at this octave (get_energy_V202).

chunk_dft = fourier_transform_V2(chunk);
local_ranges = { (double(y0):double(y1)), (double(x0):double(x1)), (double(z0):double(z1)) };

rf = [double(rf_y), double(rf_x), double(rf_z)];
y_off = double(y_offset);
x_off = double(x_offset);
z_off = double(z_offset);
yw = double(y_write_count);
xw = double(x_write_count);
zw = double(z_write_count);

[mesh_Y, mesh_X, mesh_Z] = ndgrid( ...
    linspace(1 + mod(y_off, rf(1)) / rf(1), ...
             1 + mod(y_off, rf(1)) / rf(1) + (yw - 1) / rf(1), yw), ...
    linspace(1 + mod(x_off, rf(2)) / rf(2), ...
             1 + mod(x_off, rf(2)) / rf(2) + (xw - 1) / rf(2), xw), ...
    linspace(1 + mod(z_off, rf(3)) / rf(3), ...
             1 + mod(z_off, rf(3)) / rf(3) + (zw - 1) / rf(3), zw));

radii = double(radii(:)).';
n_scales = numel(radii);
energy_chunk_4D = zeros([yw, xw, zw, n_scales]);

for s_idx = 1:n_scales
    energy_chunk_4D(:, :, :, s_idx) = interp3( ...
        energy_filter_V200( ...
            chunk_dft, matching_kernel_string, radii(s_idx), ...
            vessel_wall_thickness_in_microns, microns_per_pixel, pixels_per_sigma_PSF, ...
            local_ranges, gaussian_to_ideal_ratio, spherical_to_annular_ratio, scales_per_octave), ...
        mesh_X, mesh_Y, mesh_Z);
end

[energy, scale_idx] = min(energy_chunk_4D, [], 4);
energy(energy >= 0) = 0;
end
