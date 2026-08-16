function mesh = stretch_linspace_1based(offset, rf, count)
%STRETCH_LINSPACE_1BASED Same linspace as stretch_energy_chunk_v202 / get_energy_V202.
offset = double(offset);
rf = double(rf);
count = double(count);
mesh = linspace(1 + mod(offset, rf) / rf, ...
                1 + mod(offset, rf) / rf + (count - 1) / rf, count);
end
