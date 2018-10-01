import matplotlib.pyplot as plt
import numpy as np
import random
from pdb import set_trace as bp
import matplotlib.transforms
from matplotlib.transforms import BlendedGenericTransform
# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)


data = np.load('Data_RC.npy')
states = data[:,:14]
control = data[:,14:18]
tv_p = data[:,18:30]
time = data[:,30]
heatrate = data[:,31]
unmetHours = data[:,32]



fig = plt.figure(figsize=(12, 12))



# Remove the plot frame lines. They are unnecessary chartjunk.
ax = plt.subplot(511)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines['left'].set_position(('data',0))
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_position(('data',16))
ax.spines['bottom'].set_linewidth(2)

# Ensure that the axis ticks only show up on the bottom and left of the plot.
# Ticks on the right and top of the plot are generally unnecessary chartjunk.
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)



plt.plot(time, states[:,0], linestyle='-', linewidth = 2, color = tableau20[0], label = 'Temperature')
ax.fill_between(time, 17*np.ones(len(time)), 22*np.ones(len(time)), facecolor = tableau20[4], alpha = 0.3, label = 'Comfort Range')
lgnd = ax.legend(fontsize = 16)


plt.ylim([16, 32])
plt.xlim([0, time[-1]+20000])

labels = []
ind = []
for t in time:
    if t % 86400 == 0:
        ind.append(len(labels)*10)
        labels.append(str(t/60/24))
    else:
        labels.append(' ')
# ind.append(time[-1]+20000)
a = [18,22,26,30]
plt.yticks(a,tuple(a))
ax.tick_params(axis = 'x', width = 2, length = 10, direction = 'inout')
plt.ylabel('Room \n Temperature [$\degree$C]', fontsize = 16)
plt.grid(True)
plt.xticks(ind,[])
# plt.xticks([])
# ax.legend('Temperature', 'Comfort Range')




'''
Control Plots
'''
# Remove the plot frame lines. They are unnecessary chartjunk.
ax = plt.subplot(512)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines['left'].set_position(('data',0))
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_position(('data',16))
ax.spines['bottom'].set_linewidth(2)
# Ensure that the axis ticks only show up on the bottom and left of the plot.
# Ticks on the right and top of the plot are generally unnecessary chartjunk.
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.ylim([16, 23])
plt.xlim([0, time[-1]+20000])

plt.plot(time, control[:,3] , linestyle='-',linewidth = 2, color = tableau20[0])

plt.xticks(ind,[])
ax.tick_params(axis = 'x', width = 2, length = 10, direction = 'inout')
plt.ylabel('Heating \n Setpoint [$\degree$C]', fontsize = 16)
plt.grid(True)


ax = plt.subplot(513)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines['left'].set_position(('data',0))
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_position(('data',0))
ax.spines['bottom'].set_linewidth(2)
# Ensure that the axis ticks only show up on the bottom and left of the plot.
# Ticks on the right and top of the plot are generally unnecessary chartjunk.
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.ylim([0, 1])
plt.xlim([0, time[-1]+20000])

plt.plot(time, control[:,2] , linestyle='-',linewidth = 2, color = tableau20[0])

plt.xticks(ind,[])
ax.tick_params(axis = 'x', width = 2, length = 10, direction = 'inout')
plt.ylabel('Window \n Opening [%]', fontsize = 16)
plt.grid(True)




ax = plt.subplot(514)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines['left'].set_position(('data',0))
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_position(('data',0))
ax.spines['bottom'].set_linewidth(2)
# Ensure that the axis ticks only show up on the bottom and left of the plot.
# Ticks on the right and top of the plot are generally unnecessary chartjunk.
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.ylim([0, 1])
plt.xlim([0, time[-1]+20000])

plt.plot(time, np.round(control[:,0]) , linestyle='-',linewidth = 2, color = tableau20[0])

plt.xticks(ind,[])
ax.tick_params(axis = 'x', width = 2, length = 10, direction = 'inout')
plt.ylabel('Blind North []', fontsize = 16)
plt.grid(True)



ax = plt.subplot(515)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines['left'].set_position(('data',0))
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_position(('data',0))
ax.spines['bottom'].set_linewidth(2)
# Ensure that the axis ticks only show up on the bottom and left of the plot.
# Ticks on the right and top of the plot are generally unnecessary chartjunk.
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.ylim([0, 1])
plt.xlim([0, time[-1]+20000])

plt.plot(time, np.round(control[:,1]) , linestyle='-',linewidth = 2, color = tableau20[0])

plt.xticks(ind,('0','60', '120', '180', '240', '300', '360'))
ax.tick_params(axis = 'x', width = 2, length = 10, direction = 'inout')
plt.xlabel('Time [days]', fontsize= 14)
plt.ylabel('Blind West []', fontsize = 16)
plt.grid(True)







# plt.savefig('CompareRoom_RC.pdf',bbox_inches='tight', pad_inches=0)
plt.show()
