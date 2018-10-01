days = 10
hours = 24
minutes = 60
seconds = 60
EPTimeStep = 6 #Number of timesteps per hour in EnergyPlus

numSteps = days*hours*EPTimeStep
daystart =  86400*0/60

imagefile = 'Simulation_Data\data.pickle'

# not all zones have windows or radiators
zones_Heating = ['Coworking', 'Corridor', 'Entrance', 'LabNorth', 'LabSouth', 'MeetingSouth', 'MeetingNorth', 'Nerdroom1', 'Nerdroom2', 'RestroomM', 'RestroomW', 'Space01', 'Stairway']
# this is first equal to all avaiable zones
unused_zones = ['MeetingNorth', 'Coworking', 'MeetingSouth', 'Entrance', 'Corridor', 'LabNorth', 'LabSouth', 'Nerdroom1', 'Nerdroom2', 'RestroomM', 'RestroomW', 'Space01', 'Stairway']
zones = ['MeetingNorth', 'Coworking', 'MeetingSouth', 'Entrance', 'Corridor', 'LabNorth', 'LabSouth', 'Nerdroom1', 'Nerdroom2', 'RestroomM', 'RestroomW', 'Space01', 'Stairway']

zonenumber = {'Coworking':0, 'Corridor':1, 'Entrance':2, 'LabNorth':3, 'LabSouth':4, 'MeetingSouth':5, 'MeetingNorth':6, 'Nerdroom1':7, 'Nerdroom2':8, 'RestroomM':9 ,'RestroomW':10, 'Space01':11, 'Stairway':12}

eastSide = ['Coworking', 'Space01', 'Stairway', 'RestroomW', 'Corridor']
northSide = ['Coworking', 'MeetingNorth', 'LabNorth']
southSide = ['Nerdroom2', 'Nerdroom1', 'MeetingSouth', 'LabSouth']
westSide = ['LabSouth', 'Entrance', 'LabNorth']
