function energy = stretch_energy_filter_v200(chunk, matching_kernel_string, radius_of_lumen_in_microns, vessel_wall_thickness_in_microns, microns_per_pixel, pixels_per_sigma_PSF, y0, y1, x0, x1, z0, z1, gaussian_to_ideal_ratio, spherical_to_annular_ratio, scales_per_octave)
%STRETCH_ENERGY_FILTER_V200 MATLAB-engine Energy float body (KTD3).
% Python owns chunking, resume, and checkpoint packaging. MATLAB owns
% fourier_transform_V2 + energy_filter_V200. local_ranges are 1-based inclusive
% indices into the padded FFT grid (same convention as get_energy_V202).

chunk_dft = fourier_transform_V2(chunk);
local_ranges = { (double(y0):double(y1)), (double(x0):double(x1)), (double(z0):double(z1)) };
energy = energy_filter_V200( ...
    chunk_dft, matching_kernel_string, radius_of_lumen_in_microns, ...
    vessel_wall_thickness_in_microns, microns_per_pixel, pixels_per_sigma_PSF, ...
    local_ranges, gaussian_to_ideal_ratio, spherical_to_annular_ratio, scales_per_octave);
end
