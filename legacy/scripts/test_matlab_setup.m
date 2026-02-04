%% test_matlab_setup.m
% Simple test to verify MATLAB can find vectorize_V200 and run basic operations

fprintf('MATLAB Setup Test\n');
fprintf('=================\n');

% Check if we're in the right directory
current_dir = pwd;
fprintf('Current directory: %s\n', current_dir);

% Check if vectorize_V200 exists
if exist('vectorize_V200', 'file') == 2
    fprintf('✓ vectorize_V200.m found\n');
else
    fprintf('✗ vectorize_V200.m NOT found\n');
    fprintf('  Looking in: %s\n', current_dir);
    fprintf('  Please ensure you are in the Vectorization-Public directory\n');
end

% Check source directory
if exist('source', 'dir') == 7
    fprintf('✓ source directory found\n');
else
    fprintf('✗ source directory NOT found\n');
end

fprintf('\nTest complete.\n');
