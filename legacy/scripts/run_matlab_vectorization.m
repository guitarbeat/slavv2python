%% run_matlab_vectorization.m
% Non-interactive wrapper script for vectorize_V200
% Designed to run from MATLAB command line without user prompts
%
% Usage from MATLAB command line:
%   run_matlab_vectorization('input_file.tif', 'output_directory')
%
% Usage from Windows batch:
%   matlab -batch "cd('path/to/Vectorization-Public'); run_matlab_vectorization('input.tif', 'output_dir'); exit"

function time_stamp = run_matlab_vectorization(input_file, output_directory)
    %% Input validation
    if nargin < 2
        error('Usage: run_matlab_vectorization(input_file, output_directory)');
    end
    
    if ~exist(input_file, 'file')
        error('Input file does not exist: %s', input_file);
    end
    
    if ~exist(output_directory, 'dir')
        mkdir(output_directory);
        fprintf('Created output directory: %s\n', output_directory);
    end
    
    %% Add source directory to MATLAB path
    % This is critical - vectorize_V200 needs functions from the source directory
    current_dir = pwd;
    source_dir = fullfile(current_dir, 'source');
    if exist(source_dir, 'dir')
        addpath(source_dir);
        fprintf('Added to path: %s\n', source_dir);
    else
        warning('Source directory not found: %s', source_dir);
    end
    
    %% Display configuration
    fprintf('========================================\n');
    fprintf('SLAVV MATLAB Vectorization (Non-Interactive)\n');
    fprintf('========================================\n');
    fprintf('Input file: %s\n', input_file);
    fprintf('Output directory: %s\n', output_directory);
    fprintf('========================================\n\n');
    
    %% Set default parameters matching Python implementation
    % These match the defaults from src/slavv/utils.py validate_parameters()
    
    % Voxel size (default: isotropic 1 micron)
    microns_per_voxel = [1.0, 1.0, 1.0];
    
    % Vessel size range
    radius_of_smallest_vessel_in_microns = 1.5;
    radius_of_largest_vessel_in_microns = 50.0;
    
    % PSF parameters
    approximating_PSF = true;
    excitation_wavelength_in_microns = 1.3;  % Default, no fudge factor
    numerical_aperture = 0.95;
    sample_index_of_refraction = 1.33;
    
    % Scale parameters
    scales_per_octave = 1.5;
    gaussian_to_ideal_ratio = 1.0;  % Default (pure Gaussian)
    spherical_to_annular_ratio = 1.0;  % Default (pure spherical)
    
    % Processing parameters
    max_voxels_per_node_energy = 1e5;
    
    %% Build name-value pair inputs for vectorize_V200
    name_value_pair_inputs = { ...
        'OutputDirectory',                   output_directory, ...
        'PreviousBatch',                     'none', ...
        'PreviousWorkflow',                  'none', ...
        'StartWorkflow',                     'energy', ...
        'FinalWorkflow',                     'network', ...
        'Visual',                            'none', ...  % Skip visuals for faster comparison
        'SpecialOutput',                     {{ 'vmv', 'casx' }}, ...  % Export VMV and CASX for visualization
        'NewBatch',                          'yes', ...
        'Presumptive',                       true, ...  % Skip prompts
        'VertexCuration',                    'auto', ...  % Skip GUI curation
        'EdgeCuration',                      'auto', ...  % Skip GUI curation
        'microns_per_voxel',                 microns_per_voxel, ...
        'radius_of_smallest_vessel_in_microns', radius_of_smallest_vessel_in_microns, ...
        'radius_of_largest_vessel_in_microns', radius_of_largest_vessel_in_microns, ...
        'approximating_PSF',                 approximating_PSF, ...
        'excitation_wavelength_in_microns',  excitation_wavelength_in_microns, ...
        'numerical_aperture',                numerical_aperture, ...
        'sample_index_of_refraction',        sample_index_of_refraction, ...
        'scales_per_octave',                 scales_per_octave, ...
        'gaussian_to_ideal_ratio',          gaussian_to_ideal_ratio, ...
        'spherical_to_annular_ratio',       spherical_to_annular_ratio, ...
        'max_voxels_per_node_energy',       max_voxels_per_node_energy ...
    };
    
    %% Run vectorization
    fprintf('Starting vectorization...\n');
    tic;
    
    try
        time_stamp = vectorize_V200(input_file, name_value_pair_inputs{:});
        
        elapsed_time = toc;
        fprintf('\n========================================\n');
        fprintf('Vectorization completed successfully!\n');
        fprintf('Time stamp: %s\n', time_stamp);
        fprintf('Elapsed time: %.2f seconds\n', elapsed_time);
        fprintf('========================================\n');
        
        %% Export timing information
        % Find the batch folder that was just created
        batch_folder = fullfile(output_directory, ['batch_' time_stamp]);
        
        if exist(batch_folder, 'dir')
            timing_data = struct();
            timing_data.total_seconds = elapsed_time;
            timing_data.timestamp = time_stamp;
            timing_data.input_file = input_file;
            timing_data.matlab_version = version;
            
            % Try to extract stage-wise timing from workflow settings if available
            settings_dir = fullfile(batch_folder, 'settings');
            if exist(settings_dir, 'dir')
                workflow_files = dir(fullfile(settings_dir, 'workflow_*.mat'));
                if ~isempty(workflow_files)
                    try
                        workflow_path = fullfile(settings_dir, workflow_files(end).name);
                        workflow_data = load(workflow_path);
                        % Add workflow timing if available in the structure
                        if isfield(workflow_data, 'time_stamp')
                            timing_data.workflow_timestamps = workflow_data.time_stamp;
                        end
                    catch
                        % Ignore errors in loading workflow data
                    end
                end
            end
            
            % Save timing as JSON file
            timing_file = fullfile(batch_folder, 'timings.json');
            try
                % Convert struct to JSON string
                json_str = jsonencode(timing_data);
                
                % Write to file
                fid = fopen(timing_file, 'w');
                if fid ~= -1
                    fprintf(fid, '%s', json_str);
                    fclose(fid);
                    fprintf('Timing data saved to: %s\n', timing_file);
                end
            catch ME_json
                fprintf('Warning: Could not save timing data: %s\n', ME_json.message);
            end
        end
        
    catch ME
        elapsed_time = toc;
        fprintf('\n========================================\n');
        fprintf('ERROR: Vectorization failed!\n');
        fprintf('Elapsed time: %.2f seconds\n', elapsed_time);
        fprintf('Error message: %s\n', ME.message);
        fprintf('========================================\n');
        rethrow(ME);
    end
    
end
