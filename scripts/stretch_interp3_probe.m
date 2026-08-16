function out = stretch_interp3_probe(volume, mesh_x, mesh_y, mesh_z)
%STRETCH_INTERP3_PROBE MATLAB interp3 with get_energy_V202 argument order.
out = interp3(volume, mesh_x, mesh_y, mesh_z);
end
