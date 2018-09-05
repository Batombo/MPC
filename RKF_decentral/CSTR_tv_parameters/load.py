import matplotlib.pyplot as plt
import pickle as pl
from pdb import set_trace as bp
zones = ['MeetingNorth', 'Coworking', 'MeetingSouth', 'Entrance', 'Corridor', 'LabNorth', 'LabSouth', 'Nerdroom1', 'Nerdroom2', 'RestroomM', 'RestroomW', 'Space01', 'Stairway']
for zone in zones:
    fig_handle = pl.load(open(zone+'_test.pickle','rb'))
    fig_handle.show()
    raw_input('Press Enter to exit ...')
