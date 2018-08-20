days = 5
hours = 24
minutes = 60
seconds = 60
EPTimeStep = 6 #Number of timesteps per hour in EnergyPlus

numSteps = days*hours*EPTimeStep
# daystart =  86400*140/60
daystart =  86400*0/60

imagefile = '5days.pickle'

features = 14
numbers = 2


# not all zones have windows or radiators
zones_Heating = ['Coworking', 'Corridor', 'Entrance', 'LabNorth', 'LabSouth', 'MeetingSouth', 'MeetingNorth', 'Nerdroom1', 'Nerdroom2', 'RestroomM', 'RestroomW', 'Space01', 'Stairway']
zones = ['Coworking', 'Corridor','Elevator', 'Entrance', 'LabNorth', 'LabSouth', 'MeetingSouth', 'MeetingNorth', 'Nerdroom1', 'Nerdroom2', 'RestroomM' ,'RestroomW', 'Space01', 'Stairway']
zones_ahu = ['Coworking', 'Corridor', 'Entrance', 'LabNorth', 'LabSouth', 'MeetingSouth', 'MeetingNorth', 'Nerdroom1', 'Nerdroom2', 'RestroomW', 'Space01', 'Stairway']
