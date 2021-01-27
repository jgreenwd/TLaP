PALETTE = ((0.1, 0.5, 0.1, 1.0),   # Changeup
           (1.0, 0.7, 0.1, 0.8),   # Curveball
           (0.4, 0.0, 0.8, 0.5),   # Cutter
           (0.1, 0.1, 1.0, 1.0),   # Four-seam
           (0.1, 0.6, 0.9, 1.0),   # Two-seam
           (0.1, 0.8, 0.5, 1.0),   # Sinker
           (1.0, 0.0, 0.0, 1.0),   # Slider
           (0.0, 0.0, 0.0, 0.3),   # Splitter
           (0.8, 0.4, 0.1, 1.0),   # Knucklecurve
           (0.1, 0.5, 0.1, 0.5),   # Knuckleball
           (1.0, 0.7, 0.1, 0.4),   # Eephus
           (0.4, 0.0, 0.8, 0.25),  # Forkball
           (0.1, 0.1, 1.0, 0.5),   # Screwball
           (0.1, 0.6, 0.9, 0.5),   # Unlabeled Fastball
           (0.1, 0.8, 0.5, 0.5),   # Unlabeled Atbat
           (1.0, 0.0, 0.0, 0.5),   # Pitchout
           (0.0, 0.0, 0.0, 0.15),  # Intentional ball
           (0.8, 0.4, 0.1, 0.5))   # Unknown

WORKING_DIR = f'./'

STD_FEATURES_REMOVED = ['ab_id', 'pitcher_id', 'type_confidence', 'sz_top', 'sz_bot', 'nasty', 'zone',
                        'type_confidence', 'inning', 'x', 'y', 'x0', 'y0', 'z0', 'vx0', 'vy0', 'ax', 'ay',
                        'px', 'pz', 'start_speed', 'end_speed', 'break_length', 'break_y', 'pfx_x']

PITCH_KEYS = {'CH': 0, 'CU': 1, 'FC': 2, 'FF': 3, 'FT': 4, 'SI': 5, 'SL': 6, 'FS': 7, 'KC': 8,
              'KN': 9, 'EP': 10, 'FO': 11, 'SC': 12, 'FA': 13, 'AB': 14, 'PO': 15, 'IN': 16, 'UN': 17}

STD_PITCHES_REMOVED = ('UN', 'IN', 'PO', 'FA', 'AB', 'EP', 'FO', 'KN', 'SC', 'KC', 'FS')
