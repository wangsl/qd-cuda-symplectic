

function [] = AssLegPTest(P, RotStates, OmegaMin)

global myP

fprintf(' To read binary file: AssLegP.bin\n');
fprintf(' Rotational states: %d\n', RotStates);

binRead = fopen('AssLegP.bin');

nSize = fread(binRead, 1, 'int');

myP = cell([1, nSize]);

for i = 1 : nSize
  n = fread(binRead, 2, 'int');
  myP{i} = fread(binRead, n', 'double');
end

fclose(binRead);

if RotStates == 0 
  for i = 1 : nSize
    max(max(abs(P(:, i:end, i) - myP{i})))
  end
elseif RotStates == 1
    for i = 1 : nSize
      if mod(i-1+OmegaMin, 2) == 1
	max(max(abs(P(:, i:2:end, i) - myP{i})))
      else
	max(max(abs(P(:, i+1:2:end, i) - myP{i})))
      end
    end
elseif RotStates == 2
    for i = 1 : nSize
      if mod(i-1+OmegaMin, 2) == 0 
	max(max(abs(P(:, i:2:end, i) - myP{i})))
      else
	max(max(abs(P(:, i+1:2:end, i) - myP{i})))
      end
    end
end
