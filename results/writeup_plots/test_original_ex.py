import matplotlib as mpl
mpl.use("pgf")
pgf_with_custom_preamble = {
    'pgf.texsystem': 'pdflatex',
    'font.family': 'serif', # use serif/main font for text elements
    'text.usetex': True,    # use inline math for ticks
    'pgf.rcfonts': False,   # don't setup fonts from rc parameters
    }
mpl.rcParams.update(pgf_with_custom_preamble)

import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(6.5/2., 9/4.)) # letter: (8.5"x11"). With margins: (6.5"x9")

ax.plot([1,2,3])

label_font_size = 12
ticklabel_font_size = label_font_size * 0.8
label_pad_width = 2 # units: fraction of font-width
ax.set_xlabel('$P_\mathrm{tide}\ \mathrm{[days]}$', fontdict={'size': label_font_size}, 
              labelpad=label_pad_width)
ax.set_ylabel('$\log_{10}(Q_\star\')$', fontdict={'size': label_font_size}, 
              labelpad=label_pad_width)
for axis in [ax.xaxis, ax.yaxis]:
    for tick in axis.get_major_ticks():
            tick.label.set_fontsize(ticklabel_font_size) 

f.tight_layout()
f.savefig('foo.pdf')
