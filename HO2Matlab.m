
function [] = HO2Matlab()

global HO2Data

fprintf(' From HO2Matlab\n')

%{
if mod(HO2Data.time.steps, HO2Data.options.steps_to_copy_psi_from_device_to_host) == 0
  PlotCRP();
  if HO2Data.CRP.calculate_CRP == 1
    CRP = HO2Data.CRP;
    save(HO2Data.options.CRPMatFile, 'CRP');
  end
end
%}

if mod(HO2Data.time.steps, HO2Data.options.steps_to_copy_psi_from_device_to_host) == 0
  PlotPotWave()
end

return
