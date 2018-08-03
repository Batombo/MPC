# MPC

------------------------
			Blinds
------------------------

Shading is at the moment modeled as exterior blinds. 
	- seems like only base and first floor have exterior blinds
	- exterior blinds are modeled with a slat angle of 45° but should be modeled as 0° (see Fig 1.16 in InputOutputReference) 
		-> shades do not have slat angle ONLY blinds
		-> should be 90° but then the whole room would be dark
	- probably every floor also has interior blinds
	- if interior blinds are modeled they can also have a variable slat angle (not supported by BRCM)
	
If the reduced Model is obtained by a Neural Network Shading could be composed of
	- exterior blind with slat angle of 0° and OnIfScheduleAllows
		- possible subdivision of each window since Schedule = [0,1]
	- interior blind with variable slat angle (Schedule) and OnIfScheduleAllows
		- possible subdivision of each window since Schedule = [0,1]


------------------------
			Weather
------------------------

Use different weather files for the NN-training of the 'real' building. This means that for every location a new .idf and a new
.fmu file has to be created, due to the sizing with the different design days for different locations.

Use Berlin weather file for simulation with MPC

------------------------
			Baseboard Heaters
------------------------

Need check the following things in RKF:
	- supply water temperature
	- height width and depth of Baseboard heaters
	-> with these information HeatingDesignCapactiy can be estimated e.g. based on (https://www.hornbach.de/cms/media/de/projektbereich/raeume_1/heizungsinstallation_1/heizungsarten/Richtigen_Heizkoerper_auswaehlen.pdf)
	- for now HeatingDesignCapactiy is assumed to be 5000W

	
------------------------
			People Defintion
------------------------

Changed the field "Sensible Heat Fraction" from autosize to 1
Otherwise transmitted heat changes with different room temperatures which then leads to wrong disturbances file used in simulation 

------------------------
			FMU variable v_IG_Offices
------------------------
In BRCM this is restricted to convective heat gain
With the NN Zone Total Internal Total Heating Rate which also includes radiant heating





