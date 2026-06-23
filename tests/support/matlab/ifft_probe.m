function ifft_probe(probe_path, output_path)
    root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    addpath(genpath(fullfile(root, 'external', 'Vectorization-Public')));
    probe = jsondecode(fileread(probe_path));
    image = read_volume_tiff(char(probe.input_path));
    spacing_zyx = double(probe.microns_per_voxel_zyx(:)');
    spacing_yxz = spacing_zyx([2, 3, 1]);
    energy = probe.energy;
    chunk_dft = fourier_transform_V2(image);
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
    weights = gaussian_lengths ./ spacing_yxz;
    kernel = (weights(1) ^ 2) * (cos(2 * pi * ym) - 1);
    filtered = matching .* chunk_dft;
    spectrum = kernel .* filtered;
    output_sym = ifftn(spectrum, 'symmetric');
    output_raw = real(ifftn(spectrum));
    result = struct( ...
        'padded_shape', size(chunk_dft), ...
        'matching111', matching(2, 2, 2), ...
        'matching000', matching(1, 1, 1), ...
        'chunk000', chunk_dft(1, 1, 1), ...
        'spectrum000', spectrum(1, 1, 1), ...
        'weights', weights, ...
        'curv0_sym', output_sym(1, 1, 1), ...
        'curv0_raw', output_raw(1, 1, 1));
    save(output_path, 'result', 'spectrum', '-v7');
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