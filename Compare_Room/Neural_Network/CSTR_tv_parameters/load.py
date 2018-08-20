import matplotlib.pyplot as plt
import pickle as pl

# Load figure from disk and display
fig_handle = pl.load(open('120_7cities.pickle','rb'))
fig_handle.show()
raw_input("Press Enter to exit ...")