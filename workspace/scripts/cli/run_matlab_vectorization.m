%% run_matlab_vectorization.m
% Non-interactive, restartable wrapper script for vectorize_V200.
%
% Re-running the same command in the same output directory resumes the most
% recent matching batch from the next unfinished workflow stage.
%
% Usage from MATLAB command line:
%   run_matlab_vectorization('input_file.tif', 'output_directory')
%   run_matlab_vectorization('input_file.tif', 'output_directory', 'params.json')
%
% Usage from Windows batch:
%   matlab -wait -batch "cd('path/to/Vectorization-Public'); addpath('path/to/workspace/scripts/cli'); run_matlab_vectorization('input.tif', 'output_dir');"

function time_stamp = run_matlab_vectorization(input_file, output_directory, params_json_path)
    workflow_stages = {'energy', 'vertices', 'edges', 'network'};

    %% Input validation
    if nargin < 2
        error('Usage: run_matlab_vectorization(input_file, output_directory)');
    end

    input_file = char(string(input_file));
    output_directory = char(string(output_directory));
    if nargin < 3
        params_json_path = '';
    else
        params_json_path = char(string(params_json_path));
    end

    if ~exist(input_file, 'file')
        error('Input file does not exist: %s', input_file);
    end

    if ~exist(output_directory, 'dir')
        mkdir(output_directory);
        fprintf('Created output directory: %s\n', output_directory);
    end

    workflow_params = load_workflow_parameters(params_json_path);

    %% Add source directory to MATLAB path
    % vectorize_V200 depends on the MATLAB repository's source directory.
    current_dir = pwd;
    source_dir = fullfile(current_dir, 'source');
    if exist(source_dir, 'dir')
        addpath(source_dir);
        fprintf('Added to path: %s\n', source_dir);
    else
        warning('Source directory not found: %s', source_dir);
    end

    state_file = fullfile(output_directory, 'matlab_resume_state.json');
    resume_info = discover_resume_info(output_directory, input_file, state_file, workflow_stages);

    fprintf('========================================\n');
    fprintf('SLAVV MATLAB Vectorization (Restartable)\n');
    fprintf('========================================\n');
    fprintf('Input file: %s\n', input_file);
    fprintf('Output directory: %s\n', output_directory);
    if ~isempty(params_json_path)
        fprintf('Parameters file: %s\n', params_json_path);
    end
    if resume_info.has_batch
        fprintf('Resume batch: %s\n', resume_info.batch_timestamp);
    else
        fprintf('Resume batch: (new batch)\n');
    end
    fprintf('========================================\n\n');

    batch_timestamp = '';
    last_completed_stage = '';
    stage_seconds = struct();

    if resume_info.has_batch
        batch_timestamp = resume_info.batch_timestamp;
        last_completed_stage = resume_info.last_completed_stage;
        time_stamp = batch_timestamp;

        if isempty(resume_info.next_stage)
            fprintf('Existing batch_%s is already complete. Nothing to do.\n', batch_timestamp);
            write_resume_state( ...
                state_file, ...
                input_file, ...
                output_directory, ...
                batch_timestamp, ...
                last_completed_stage, ...
                'completed' ...
            );
            return;
        end

        fprintf( ...
            'Resuming batch_%s from stage: %s\n', ...
            batch_timestamp, ...
            resume_info.next_stage ...
        );
        stage_start_index = resume_info.next_stage_index;
    else
        fprintf('Starting a new batch and running all workflow stages.\n');
        stage_start_index = 1;
        time_stamp = '';
    end

    total_timer = tic;

    for stage_index = stage_start_index:numel(workflow_stages)
        stage_name = workflow_stages{stage_index};
        fprintf('\n----------------------------------------\n');
        fprintf('Running stage %d/%d: %s\n', stage_index, numel(workflow_stages), stage_name);
        fprintf('----------------------------------------\n');

        write_resume_state( ...
            state_file, ...
            input_file, ...
            output_directory, ...
            batch_timestamp, ...
            last_completed_stage, ...
            ['running:', stage_name] ...
        );

        name_value_pair_inputs = build_stage_inputs( ...
            output_directory, ...
            stage_name, ...
            batch_timestamp, ...
            workflow_params ...
        );

        stage_timer = tic;

        try
            if isempty(batch_timestamp)
                workflow_timestamp = vectorize_V200(input_file, name_value_pair_inputs{:});
            else
                workflow_timestamp = vectorize_V200(name_value_pair_inputs{:});
            end
        catch ME
            elapsed_time = toc(total_timer);
            fprintf('\n========================================\n');
            fprintf('ERROR: Stage %s failed!\n', stage_name);
            fprintf('Elapsed time: %.2f seconds\n', elapsed_time);
            fprintf('Error message: %s\n', ME.message);
            fprintf('========================================\n');
            rethrow(ME);
        end

        stage_elapsed = toc(stage_timer);
        stage_seconds.(stage_name) = stage_elapsed;

        if isempty(batch_timestamp)
            batch_timestamp = workflow_timestamp;
            time_stamp = batch_timestamp;
        end

        last_completed_stage = stage_name;
        fprintf('Stage %s completed in %.2f seconds.\n', stage_name, stage_elapsed);

        write_resume_state( ...
            state_file, ...
            input_file, ...
            output_directory, ...
            batch_timestamp, ...
            last_completed_stage, ...
            'stage-completed' ...
        );
    end

    total_elapsed = toc(total_timer);

    write_resume_state( ...
        state_file, ...
        input_file, ...
        output_directory, ...
        batch_timestamp, ...
        last_completed_stage, ...
        'completed' ...
    );
    export_timing_information( ...
        output_directory, ...
        batch_timestamp, ...
        input_file, ...
        total_elapsed, ...
        stage_seconds ...
    );

    fprintf('\n========================================\n');
    fprintf('Vectorization completed successfully!\n');
    fprintf('Batch timestamp: %s\n', batch_timestamp);
    fprintf('Elapsed time: %.2f seconds\n', total_elapsed);
    fprintf('========================================\n');
end

function name_value_pair_inputs = build_stage_inputs(output_directory, stage_name, batch_timestamp, workflow_params)
    energy_inputs = { ...
        'microns_per_voxel', workflow_params.microns_per_voxel, ...
        'radius_of_smallest_vessel_in_microns', workflow_params.radius_of_smallest_vessel_in_microns, ...
        'radius_of_largest_vessel_in_microns', workflow_params.radius_of_largest_vessel_in_microns, ...
        'approximating_PSF', workflow_params.approximating_PSF, ...
        'excitation_wavelength_in_microns', workflow_params.excitation_wavelength_in_microns, ...
        'numerical_aperture', workflow_params.numerical_aperture, ...
        'sample_index_of_refraction', workflow_params.sample_index_of_refraction, ...
        'scales_per_octave', workflow_params.scales_per_octave, ...
        'gaussian_to_ideal_ratio', workflow_params.gaussian_to_ideal_ratio, ...
        'spherical_to_annular_ratio', workflow_params.spherical_to_annular_ratio, ...
        'max_voxels_per_node_energy', workflow_params.max_voxels_per_node_energy ...
    };

    vertex_inputs = { ...
        'space_strel_apothem', workflow_params.space_strel_apothem, ...
        'energy_upper_bound', workflow_params.energy_upper_bound, ...
        'max_voxels_per_node', workflow_params.max_voxels_per_node, ...
        'length_dilation_ratio', workflow_params.length_dilation_ratio ...
    };

    edge_inputs = { ...
        'number_of_edges_per_vertex', workflow_params.number_of_edges_per_vertex ...
    };

    common_inputs = { ...
        'OutputDirectory', output_directory, ...
        'Visual', 'none', ...
        'Presumptive', true, ...
        'Forgetful', 'none' ...
    };

    switch stage_name
        case 'energy'
            workflow_specific_inputs = energy_inputs;
            stage_specific_inputs = { ...
                'FinalWorkflow', 'one', ...
                'SpecialOutput', 'none', ...
                'VertexCuration', 'none', ...
                'EdgeCuration', 'none' ...
            };
        case 'vertices'
            workflow_specific_inputs = [energy_inputs, vertex_inputs];
            stage_specific_inputs = { ...
                'FinalWorkflow', 'one', ...
                'SpecialOutput', 'none', ...
                'VertexCuration', 'auto', ...
                'EdgeCuration', 'none' ...
            };
        case 'edges'
            workflow_specific_inputs = [energy_inputs, vertex_inputs, edge_inputs];
            stage_specific_inputs = { ...
                'FinalWorkflow', 'one', ...
                'SpecialOutput', 'none', ...
                'VertexCuration', 'none', ...
                'EdgeCuration', 'auto' ...
            };
        case 'network'
            workflow_specific_inputs = [energy_inputs, vertex_inputs, edge_inputs];
            stage_specific_inputs = { ...
                'FinalWorkflow', 'one', ...
                'SpecialOutput', 'none', ...
                'VertexCuration', 'none', ...
                'EdgeCuration', 'none' ...
            };
        otherwise
            error('Unsupported stage: %s', stage_name);
    end

    if isempty(batch_timestamp)
        name_value_pair_inputs = [ ...
            { ...
                'PreviousBatch', 'none', ...
                'PreviousWorkflow', 'none', ...
                'StartWorkflow', 'energy', ...
                'NewBatch', 'yes' ...
            }, ...
            stage_specific_inputs, ...
            workflow_specific_inputs, ...
            common_inputs ...
        ];
    else
        name_value_pair_inputs = [ ...
            { ...
                'PreviousBatch', batch_timestamp, ...
                'PreviousWorkflow', 'recent', ...
                'StartWorkflow', stage_name, ...
                'NewBatch', 'no' ...
            }, ...
            stage_specific_inputs, ...
            workflow_specific_inputs, ...
            common_inputs ...
        ];
    end
end

function workflow_params = load_workflow_parameters(params_json_path)
    workflow_params = default_workflow_parameters();

    if isempty(params_json_path)
        return;
    end

    if ~exist(params_json_path, 'file')
        warning('Parameters file not found: %s', params_json_path);
        return;
    end

    try
        raw = fileread(params_json_path);
        if isempty(strtrim(raw))
            return;
        end
        loaded_params = jsondecode(raw);
    catch ME
        warning('Failed to parse parameters file %s: %s', params_json_path, ME.message);
        return;
    end

    supported_fields = fieldnames(workflow_params);
    loaded_fields = fieldnames(loaded_params);
    unsupported_fields = setdiff(loaded_fields, supported_fields);
    if ~isempty(unsupported_fields)
        fprintf( ...
            'Ignoring MATLAB-unsupported comparison parameter(s): %s\n', ...
            strjoin(unsupported_fields, ', ') ...
        );
    end

    for field_index = 1:numel(supported_fields)
        field_name = supported_fields{field_index};
        if isfield(loaded_params, field_name)
            workflow_params.(field_name) = loaded_params.(field_name);
        end
    end

    if isnumeric(workflow_params.microns_per_voxel)
        workflow_params.microns_per_voxel = double(workflow_params.microns_per_voxel(:)');
    end
end

function workflow_params = default_workflow_parameters()
    workflow_params = struct();
    workflow_params.microns_per_voxel = [1.0, 1.0, 1.0];
    workflow_params.radius_of_smallest_vessel_in_microns = 1.5;
    workflow_params.radius_of_largest_vessel_in_microns = 50.0;
    workflow_params.approximating_PSF = true;
    workflow_params.excitation_wavelength_in_microns = 1.3;
    workflow_params.numerical_aperture = 0.95;
    workflow_params.sample_index_of_refraction = 1.33;
    workflow_params.scales_per_octave = 1.5;
    workflow_params.gaussian_to_ideal_ratio = 1.0;
    workflow_params.spherical_to_annular_ratio = 1.0;
    workflow_params.max_voxels_per_node_energy = 1e5;
    workflow_params.space_strel_apothem = 1;
    workflow_params.energy_upper_bound = 0;
    workflow_params.max_voxels_per_node = 6000;
    workflow_params.length_dilation_ratio = 1;
    workflow_params.number_of_edges_per_vertex = 4;
end

function resume_info = discover_resume_info(output_directory, input_file, state_file, workflow_stages)
    resume_info = struct( ...
        'has_batch', false, ...
        'batch_timestamp', '', ...
        'batch_folder', '', ...
        'last_completed_stage', '', ...
        'next_stage', 'energy', ...
        'next_stage_index', 1 ...
    );

    state = read_resume_state(state_file);
    candidate_batch_folder = '';

    if isstruct(state) ...
            && isfield(state, 'input_file') ...
            && strcmp(normalize_compare_path(state.input_file), normalize_compare_path(input_file)) ...
            && isfield(state, 'batch_timestamp') ...
            && ~isempty(char(string(state.batch_timestamp)))
        candidate_batch_folder = fullfile( ...
            output_directory, ...
            ['batch_', char(string(state.batch_timestamp))] ...
        );
        if ~batch_matches_input(candidate_batch_folder, input_file)
            candidate_batch_folder = '';
        end
    end

    if isempty(candidate_batch_folder)
        batch_listing = dir(fullfile(output_directory, 'batch_*-*'));
        batch_listing = batch_listing([batch_listing.isdir]);
        if ~isempty(batch_listing)
            [~, order] = sort({batch_listing.name});
            batch_listing = batch_listing(order);
            for batch_index = numel(batch_listing):-1:1
                batch_folder = fullfile(output_directory, batch_listing(batch_index).name);
                if batch_matches_input(batch_folder, input_file)
                    candidate_batch_folder = batch_folder;
                    break;
                end
            end
        end
    end

    if isempty(candidate_batch_folder)
        return;
    end

    last_completed_stage = detect_completed_stage(candidate_batch_folder);
    next_stage_index = index_after_stage(last_completed_stage, workflow_stages);

    resume_info.has_batch = true;
    resume_info.batch_folder = candidate_batch_folder;
    resume_info.batch_timestamp = extract_batch_timestamp(candidate_batch_folder);
    resume_info.last_completed_stage = last_completed_stage;

    if next_stage_index > numel(workflow_stages)
        resume_info.next_stage = '';
        resume_info.next_stage_index = numel(workflow_stages) + 1;
    else
        resume_info.next_stage = workflow_stages{next_stage_index};
        resume_info.next_stage_index = next_stage_index;
    end
end

function completed_stage = detect_completed_stage(batch_folder)
    completed_stage = '';
    [~, roi_names] = load_batch_settings(batch_folder);

    if isempty(roi_names)
        return;
    end

    if stage_outputs_exist(batch_folder, roi_names, 'energy')
        completed_stage = 'energy';
    end

    if stage_outputs_exist(batch_folder, roi_names, 'vertices')
        completed_stage = 'vertices';
    end

    if stage_outputs_exist(batch_folder, roi_names, 'edges')
        completed_stage = 'edges';
    end

    if stage_outputs_exist(batch_folder, roi_names, 'network')
        completed_stage = 'network';
    end
end

function exists_all = stage_outputs_exist(batch_folder, roi_names, stage_name)
    if isempty(roi_names)
        exists_all = false;
        return;
    end

    switch stage_name
        case 'energy'
            target_dir = fullfile(batch_folder, 'data');
            prefixes = {'energy_'};
        case 'vertices'
            target_dir = fullfile(batch_folder, 'vectors');
            prefixes = {'vertices_', 'curated_vertices_'};
        case 'edges'
            target_dir = fullfile(batch_folder, 'vectors');
            prefixes = {'edges_', 'curated_edges_'};
        case 'network'
            target_dir = fullfile(batch_folder, 'vectors');
            prefixes = {'network_'};
        otherwise
            exists_all = false;
            return;
    end

    exists_all = true;
    for roi_index = 1:numel(roi_names)
        roi_name = char(string(roi_names{roi_index}));
        for prefix_index = 1:numel(prefixes)
            pattern = [prefixes{prefix_index}, '*', roi_name, '.mat'];
            listing = dir(fullfile(target_dir, pattern));
            if isempty(listing)
                exists_all = false;
                return;
            end
        end
    end
end

function matches = batch_matches_input(batch_folder, input_file)
    matches = false;
    if ~exist(batch_folder, 'dir')
        return;
    end

    [optional_inputs, ~] = load_batch_settings(batch_folder);
    if isempty(optional_inputs)
        return;
    end

    normalized_input = normalize_compare_path(input_file);
    for input_index = 1:numel(optional_inputs)
        candidate = normalize_compare_path(optional_inputs{input_index});
        if strcmp(candidate, normalized_input)
            matches = true;
            return;
        end
    end
end

function [optional_inputs, roi_names] = load_batch_settings(batch_folder)
    optional_inputs = {};
    roi_names = {};

    settings_dir = fullfile(batch_folder, 'settings');
    batch_file = fullfile(settings_dir, 'batch.mat');
    if ~exist(batch_file, 'file')
        batch_file = fullfile(settings_dir, 'batch');
    end

    if ~exist(batch_file, 'file')
        return;
    end

    try
        batch_settings = load(batch_file, 'optional_input', 'ROI_names');
    catch
        return;
    end

    if isfield(batch_settings, 'ROI_names')
        roi_names = ensure_cellstr(batch_settings.ROI_names);
    end

    if isfield(batch_settings, 'optional_input')
        optional_inputs = ensure_cellstr(batch_settings.optional_input);
    end
end

function values = ensure_cellstr(raw_value)
    if iscell(raw_value)
        values = cellfun(@(x) char(string(x)), raw_value, 'UniformOutput', false);
        return;
    end

    if isstring(raw_value)
        values = cellstr(raw_value);
        return;
    end

    values = {char(string(raw_value))};
end

function index = index_after_stage(last_completed_stage, workflow_stages)
    if isempty(last_completed_stage)
        index = 1;
        return;
    end

    stage_index = find(strcmp(workflow_stages, last_completed_stage), 1);
    if isempty(stage_index)
        index = 1;
        return;
    end

    index = stage_index + 1;
end

function batch_timestamp = extract_batch_timestamp(batch_folder)
    [~, batch_name] = fileparts(batch_folder);
    batch_timestamp = '';
    if startsWith(batch_name, 'batch_')
        batch_timestamp = batch_name(7:end);
    end
end

function write_resume_state( ...
    state_file, ...
    input_file, ...
    output_directory, ...
    batch_timestamp, ...
    last_completed_stage, ...
    status)
    state = struct();
    state.input_file = input_file;
    state.output_directory = output_directory;
    state.batch_timestamp = batch_timestamp;
    if isempty(batch_timestamp)
        state.batch_folder = '';
    else
        state.batch_folder = fullfile(output_directory, ['batch_', batch_timestamp]);
    end
    state.last_completed_stage = last_completed_stage;
    state.status = status;
    state.updated_at = datestr(now, 31);
    save_json(state_file, state);
end

function state = read_resume_state(state_file)
    state = struct();
    if ~exist(state_file, 'file')
        return;
    end

    try
        raw = fileread(state_file);
        if isempty(strtrim(raw))
            return;
        end
        state = jsondecode(raw);
    catch
        state = struct();
    end
end

function save_json(file_path, data_struct)
    file_id = fopen(file_path, 'w');
    if file_id == -1
        warning('Could not write JSON file: %s', file_path);
        return;
    end

    cleanup_obj = onCleanup(@() fclose(file_id));
    fprintf(file_id, '%s', jsonencode(data_struct));
    clear cleanup_obj
end

function export_timing_information( ...
    output_directory, ...
    batch_timestamp, ...
    input_file, ...
    total_elapsed, ...
    stage_seconds)
    if isempty(batch_timestamp)
        return;
    end

    batch_folder = fullfile(output_directory, ['batch_', batch_timestamp]);
    if ~exist(batch_folder, 'dir')
        return;
    end

    timing_data = struct();
    timing_data.total_seconds = total_elapsed;
    timing_data.timestamp = batch_timestamp;
    timing_data.input_file = input_file;
    timing_data.matlab_version = version;
    timing_data.wrapper_mode = 'staged-resume';
    timing_data.stage_seconds = stage_seconds;

    timing_file = fullfile(batch_folder, 'timings.json');
    save_json(timing_file, timing_data);
    fprintf('Timing data saved to: %s\n', timing_file);
end

function normalized = normalize_compare_path(path_value)
    normalized = char(string(path_value));
    normalized = strrep(normalized, '/', filesep);
    normalized = strrep(normalized, '\', filesep);

    try
        normalized = char(java.io.File(normalized).getCanonicalPath());
    catch
        normalized = char(string(normalized));
    end

    if ispc
        normalized = lower(normalized);
    end
end
