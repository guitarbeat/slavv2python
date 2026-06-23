function export_random_matching_reference(manifest_path, output_path)
    root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    addpath(genpath(fullfile(root, 'external', 'Vectorization-Public')));
    payload = jsondecode(fileread(manifest_path));
    energy = payload.energy;
    cases = payload.cases;
    seen = containers.Map('KeyType', 'char', 'ValueType', 'logical');
    records = struct('spacing_yxz', {}, 'shape_yxz', {}, 'values', {});
    for ci = 1:numel(cases)
        current = cases(ci);
        spacing_zyx = double(current.microns_per_voxel_zyx(:)');
        spacing_yxz = spacing_zyx([2, 3, 1]);
        key = sprintf('%.17g,%.17g,%.17g', spacing_yxz);
        if isKey(seen, key)
            continue;
        end
        seen(key) = true;
        image = read_volume_tiff(char(current.input_path));
        chunk_dft = fourier_transform_V2(image);
        matching = matching_kernel_dft(chunk_dft, spacing_yxz, energy);
        records(end + 1) = struct( ...
            'spacing_yxz', spacing_yxz, ...
            'shape_yxz', size(chunk_dft), ...
            'values', matching(:)');
    end
    results = struct('records', records);
    save(output_path, 'results', '-v7');
end

function matching = matching_kernel_dft(chunk_dft, spacing_yxz, energy)
    dims = size(chunk_dft);
    [ym, xm, zm] = ndgrid( ...
        [0:dims(1) / 2 - 1, -dims(1) / 2:-1] / dims(1), ...
        [0:dims(2) / 2 - 1, -dims(2) / 2:-1] / dims(2), ...
        [0:dims(3) / 2 - 1, -dims(3) / 2:-1] / dims(3));
    ymic = ym / spacing_yxz(1); xmic = xm / spacing_yxz(2); zmic = zm / spacing_yxz(3);
    radius = double(energy.radius_microns);
    psf = double(energy.pixels_per_sigma_psf_yxz(:)');
    g2i = double(energy.gaussian_to_ideal_ratio);
    s2a = double(energy.spherical_to_annular_ratio);
    gaussian_lengths = g2i * radius + [0, 0, 0];
    pulse_sq = (1 - g2i ^ 2) * radius ^ 2 + (psf .* spacing_yxz) .^ 2;
    radial = 2 * pi * sqrt(ymic .^ 2 * pulse_sq(1) + xmic .^ 2 * pulse_sq(2) + zmic .^ 2 * pulse_sq(3));
    spherical = (pi / 2 ./ radial) .^ 0.5 .* (besselj(2.5, radial) + besselj(0.5, radial));
    spherical(radial == 0) = 1;
    gaussian = exp(-2 * pi ^ 2 * ((ymic * gaussian_lengths(1)) .^ 2 + ...
        (xmic * gaussian_lengths(2)) .^ 2 + (zmic * gaussian_lengths(3)) .^ 2));
    annular = cos(radial);
    matching = gaussian .* ((1 - s2a) * annular + s2a * spherical);
end

function image = read_volume_tiff(tif_path)
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