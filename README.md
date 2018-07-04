# MPC

------------------------
			Blinds
------------------------

Shading is at the moment modeled as exterior blinds. 
	- seems like only base and first floor have exterior blinds
	- exterior blinds are modeled with a slat angle of 45° but should be modeled as 0° (see Fig 1.16 in InputOutputReference) 
		-> should be 90° but then the whole room would be dark
	- probably every floor also has interior blinds
	- if interior blinds are modeled they can also have a variable slat angle (not supported by BRCM)
	
If the reduced Model is obtained by a Neural Network Shading could be composed of
	- exterior blind with slat angle of 0° and OnIfScheduleAllows
		- possible subdivision of each window since Schedule = [0,1]
	- interior blind with variable slat angle (Schedule) and OnIfScheduleAllows
		- possible subdivision of each window since Schedule = [0,1]


			