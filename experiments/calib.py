# Generically useful definitions and info for calibration plots
#
# Kiri Wagstaff
# January 15, 2021

# Names, colors, and markers for plotting
# method: (full_name, color, marker, short_name)
calib_info = {'uncal': ('Uncalibrated', 'red', 'o', 'Uncal.'),
              'platt': ('Platt scaling', 'brown', '^', 'Platt'),
              'ts': ('Temp. scaling', 'brown', '^', 'TS'),
              'hist': ('Hist. binning', 'sandybrown', '>', 'Hist bin'),
              'scalebin': ('Platt binning', 'brown', '<', 'Scale bin'),
              'retrain': ('Retrain', 'pink', 's'),
              'sba': ('SBA', 'darkorchid', 'd', 'SBA'),
              'swc': ('SWC', 'dodgerblue', 'v', 'SWC'),
              'swc-hh': ('SWC-HH', 'limegreen', '>', 'SWC-HH'),
              # Platt (or TS), then SWC
              'platt-swc': ('Platt+SWC', 'limegreen', '*', 'Platt+SWC'),
              'ts-swc': ('TS+SWC', 'limegreen', '*', 'TS+SWC'),
}

# Formal names for data sets
ds_name = {'cifar10': 'CIFAR-10',
           'cifar100': 'CIFAR-100',
           'covid': 'COVID-19 diagnosis',
           'credit': 'credit',
           'ctg': 'cardiotography',
           'letter': 'letter',
           'fashion-mnist': 'fashion-mnist',
           'mnist-1v7': 'mnist-1v7',
           'mnist-3v5': 'mnist-3v5',
           'mnist-3v8': 'mnist-3v8',
           'mnist-4v9': 'mnist-4v9',
           'mnist10': 'mnist10',
           'moons': 'moons'}
