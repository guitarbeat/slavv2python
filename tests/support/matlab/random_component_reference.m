% Generate MATLAB R2019a references for the seeded random-component corpus.
%
% The Python runner materializes uint16 TIFFs and a JSON manifest first. This
% driver deliberately uses no MATLAB RNG, so every value is attributable to a
% manifest seed and can be reproduced on the self-hosted MATLAB runner.
function random_component_reference(manifest_path, output_path, mode)
    root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    addpath(genpath(fullfile(root, 'external', 'Vectorization-Public')));
    payload = jsondecode(fileread(manifest_path));
    if nargin < 3 || isempty(mode)
        mode = 'structural';
    end
    include_hessian = strcmpi(mode, 'diagnostics');

    contexts = payload.linspace_contexts;
    linspace_records = repmat(struct('offset', 0, 'stride', 0, 'count', 0, ...
        'local_start', 0, 'values', []), 1, numel(contexts));
    for i = 1:numel(contexts)
        ctx = contexts(i);
        start_1based = 1 + double(ctx.offset) / double(ctx.stride) - double(ctx.local_start);
        stop_1based = start_1based + (double(ctx.count) - 1) / double(ctx.stride);
        linspace_records(i).offset = double(ctx.offset);
        linspace_records(i).stride = double(ctx.stride);
        linspace_records(i).count = double(ctx.count);
        linspace_records(i).local_start = double(ctx.local_start);
        linspace_records(i).values = linspace(start_1based, stop_1based, double(ctx.count)) - 1;
    end

    cases = payload.cases;
    sample_template = struct('coordinate_yxz', [], 'curvatures', [], 'gradient', [], ...
        'laplacian', 0, 'valid', false, 'energy', 0, 'winner_scale', 0);
    case_records = repmat(struct('case_id', '', 'interpolation', [], ...
        'padded_shape_yxz', [], 'samples', sample_template), 1, numel(cases));
    for ci = 1:numel(cases)
        current = cases(ci);
        image = read_volume_tiff(char(current.input_path));
        queries = double(current.query_yxz);
        interpolation = interp3(image, queries(:, 2) + 1, queries(:, 1) + 1, ...
            queries(:, 3) + 1, 'linear', 0);
        case_records(ci).case_id = char(current.id);
        case_records(ci).interpolation = interpolation(:)';
        [padded_shape, samples] = energy_samples(image, current, payload.energy, sample_template, include_hessian);
        case_records(ci).padded_shape_yxz = padded_shape;
        case_records(ci).samples = samples;
    end

    results = struct('linspace_records', linspace_records, 'case_records', case_records);
    save(output_path, 'results', '-v7');
end

function [padded_shape, samples] = energy_samples(image, current, energy, sample_template, include_hessian)
    spacing_zyx = double(current.microns_per_voxel_zyx(:)');
    spacing_yxz = spacing_zyx([2, 3, 1]);
    radius = double(energy.radius_microns);
    psf = double(energy.pixels_per_sigma_psf_yxz(:)');
    g2i = double(energy.gaussian_to_ideal_ratio);
    s2a = double(energy.spherical_to_annular_ratio);
    chunk_dft = fourier_transform_V2(image);
    padded_shape = double(size(chunk_dft));
    if ~include_hessian
        samples = repmat(sample_template, 1, 0);
        return;
    end
    local_ranges = {1:size(chunk_dft, 1), 1:size(chunk_dft, 2), 1:size(chunk_dft, 3)};
    coords = [0, 0, 0; 1, 2, 3; ...
        min(7, size(image, 1) - 1), min(15, size(image, 2) - 1), min(7, size(image, 3) - 1); ...
        size(image, 1) - 1, size(image, 2) - 1, size(image, 3) - 1];
    samples = repmat(sample_template, 1, size(coords, 1));
    for i = 1:size(coords, 1)
        y = coords(i, 1) + 1;
        x = coords(i, 2) + 1;
        z = coords(i, 3) + 1;
        derivative = derivatives_at(chunk_dft, radius, spacing_yxz, psf, local_ranges, g2i, s2a, y, x, z);
        samples(i).coordinate_yxz = coords(i, :);
        samples(i).curvatures = derivative.curvatures;
        samples(i).gradient = derivative.gradient;
        samples(i).laplacian = derivative.laplacian;
        samples(i).valid = derivative.laplacian < 0;
        if samples(i).valid
            samples(i).energy = principal_energy_from_derivatives(derivative.gradient, derivative.curvatures);
        else
            samples(i).energy = inf;
        end
        samples(i).winner_scale = 0;
    end
end

function deriv = derivatives_at(chunk_dft, radius, spacing, psf, local_ranges, g2i, s2a, y, x, z)
    dims = size(chunk_dft);
    [ym, xm, zm] = ndgrid( ...
        [0:dims(1) / 2 - 1, -dims(1) / 2:-1] / dims(1), ...
        [0:dims(2) / 2 - 1, -dims(2) / 2:-1] / dims(2), ...
        [0:dims(3) / 2 - 1, -dims(3) / 2:-1] / dims(3));
    ymic = ym / spacing(1); xmic = xm / spacing(2); zmic = zm / spacing(3);
    gaussian_lengths = g2i * radius + [0, 0, 0];
    pulse_sq = (1 - g2i ^ 2) * radius ^ 2 + (psf .* spacing) .^ 2;
    radial = 2 * pi * sqrt(ymic .^ 2 * pulse_sq(1) + xmic .^ 2 * pulse_sq(2) + zmic .^ 2 * pulse_sq(3));
    spherical = (pi / 2 ./ radial) .^ 0.5 .* (besselj(2.5, radial) + besselj(0.5, radial));
    spherical(radial == 0) = 1;
    gaussian = exp(-2 * pi ^ 2 * ((ymic * gaussian_lengths(1)) .^ 2 + ...
        (xmic * gaussian_lengths(2)) .^ 2 + (zmic * gaussian_lengths(3)) .^ 2));
    annular = cos(radial);
    matching = gaussian .* ((1 - s2a) * annular + s2a * spherical);
    weights = gaussian_lengths ./ spacing;
    curvatures = zeros(1, 6); gradient = zeros(1, 3);
    curvatures(1) = sample_ifft((weights(1) ^ 2) * (cos(2 * pi * ym) - 1), matching, chunk_dft, local_ranges, y, x, z);
    curvatures(2) = sample_ifft((weights(2) ^ 2) * (cos(2 * pi * xm) - 1), matching, chunk_dft, local_ranges, y, x, z);
    curvatures(3) = sample_ifft((weights(3) ^ 2) * (cos(2 * pi * zm) - 1), matching, chunk_dft, local_ranges, y, x, z);
    curvatures(4) = sample_ifft(weights(1) * weights(2) * (cos(2 * pi * sqrt(abs(ym .* xm))) - 1) .* sign(ym .* xm) / 4, matching, chunk_dft, local_ranges, y, x, z);
    curvatures(5) = sample_ifft(weights(2) * weights(3) * (cos(2 * pi * sqrt(abs(xm .* zm))) - 1) .* sign(xm .* zm) / 4, matching, chunk_dft, local_ranges, y, x, z);
    curvatures(6) = sample_ifft(weights(3) * weights(1) * (cos(2 * pi * sqrt(abs(zm .* ym))) - 1) .* sign(zm .* ym) / 4, matching, chunk_dft, local_ranges, y, x, z);
    gradient(1) = sample_ifft(1i * weights(1) * sin(2 * pi * ym) / 2, matching, chunk_dft, local_ranges, y, x, z);
    gradient(2) = sample_ifft(1i * weights(2) * sin(2 * pi * xm) / 2, matching, chunk_dft, local_ranges, y, x, z);
    gradient(3) = sample_ifft(1i * weights(3) * sin(2 * pi * zm) / 2, matching, chunk_dft, local_ranges, y, x, z);
    deriv = struct('curvatures', curvatures, 'gradient', gradient, 'laplacian', sum(curvatures(1:3)));
end

function energy_value = principal_energy_from_derivatives(gradient, curvatures)
    hessian = [curvatures(1), curvatures(4), curvatures(6); ...
               curvatures(4), curvatures(2), curvatures(5); ...
               curvatures(6), curvatures(5), curvatures(3)];
    [vectors, values] = eig(hessian);
    principal_values = diag(values)';
    projections = gradient * vectors;
    principal_energies = zeros(1, 3);
    for index = 1:3
        principal_energies(index) = principal_values(index) ...
            * exp(-(projections(index) / principal_values(index)) ^ 2 / 2);
    end
    principal_energies(3) = min(principal_energies(3), 0);
    energy_value = sum(principal_energies);
    if ~isfinite(energy_value) || energy_value >= 0
        energy_value = inf;
    end
end

function value = sample_ifft(kernel, matching, chunk_dft, local_ranges, y, x, z)
    output = ifftn(kernel .* matching .* chunk_dft, 'symmetric');
    output = output(local_ranges{1}, local_ranges{2}, local_ranges{3});
    value = output(y, x, z);
end

function image = read_volume_tiff(tif_path)
    % R2019a-compatible multi-page TIFF reader (tiffreadVolume is unavailable).
    info = imfinfo(tif_path);
    number_of_slices = numel(info);
    slice_height = info(1).Height;
    slice_width = info(1).Width;
    image = zeros(slice_height, slice_width, number_of_slices);
    for slice_index = 1:number_of_slices
        image(:, :, slice_index) = imread(tif_path, slice_index, 'Info', info);
    end
    image = double(image);
end
