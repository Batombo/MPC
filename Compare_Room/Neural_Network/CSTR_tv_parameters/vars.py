days = 2
hours = 24
minutes = 60
seconds = 60
EPTimeStep = 6 #Number of timesteps per hour in EnergyPlus

numSteps = days*hours*EPTimeStep
# daystart =  86400*140/60
daystart =  86400*96/60

imagefile = 'test.pickle'
