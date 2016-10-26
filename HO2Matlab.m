
function [] = HO2Matlab()

global HO2Data

fprintf(' From HO2Matlab\n')

if mod(HO2Data.time.steps, HO2Data.options.steps_to_copy_psi_from_device_to_host) == 0
  PlotCRP();
end

if mod(HO2Data.time.steps, HO2Data.options.steps_to_copy_psi_from_device_to_host) == 0
  PlotPotWave()
end

return
