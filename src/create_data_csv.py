# This script creates a CSV with columns:
# 1. Date stamp from filename
# 2. Date stamp from DATE-OBS in the FITS header
# 3. Path to HMI file
# 4. Path to AIA file

import os
import csv
from astropy.io import fits

solar_data_dir = os.path.join(os.path.dirname(__file__), '../solar_data')
aia_dir = os.path.join(solar_data_dir, 'aia_images')
hmi_dir = os.path.join(solar_data_dir, 'hmi_images')

# Helper to extract and normalize date stamp from filename (AIA and HMI)
import re
def extract_date_from_filename(filename):
    # For AIA: aia.lev1_euv_12s.2023-12-31T235930Z.193.image_lev1.fits
    # For HMI: hmi.m_720s.20240101_000000_TAI.3.magnetogram.fits
    if 'aia' in filename:
        m = re.search(r'(\d{4}-\d{2}-\d{2})T(\d{6})Z', filename)
        if m:
            # Convert to YYYYMMDD_HHMMSS
            return m.group(1).replace('-', '') + '_' + m.group(2)
    elif 'hmi' in filename:
        m = re.search(r'(\d{8}_\d{6})', filename)
        if m:
            return m.group(1)
    return ''

# Find all AIA and HMI files
aia_files = sorted([f for f in os.listdir(aia_dir) if f.endswith('.fits') and 'image_lev1' in f])
hmi_files = sorted([f for f in os.listdir(hmi_dir) if f.endswith('.fits') and 'magnetogram' in f])

# Build a dict for quick lookup by date stamp
from datetime import datetime, timedelta

# Helper to parse normalized date string to datetime
def parse_date(date_str):
    # date_str: 'YYYYMMDD_HHMMSS'
    return datetime.strptime(date_str, '%Y%m%d_%H%M%S')

# Build lists of (date, path)
aia_list = [(extract_date_from_filename(f), os.path.join(aia_dir, f)) for f in aia_files]
hmi_list = [(extract_date_from_filename(f), os.path.join(hmi_dir, f)) for f in hmi_files]

# Remove entries with empty date
aia_list = [(d, p) for d, p in aia_list if d]
hmi_list = [(d, p) for d, p in hmi_list if d]

# Sort by date
aia_list.sort()
hmi_list.sort()

# For each AIA, find closest HMI within 1 minute
time_tolerance = timedelta(minutes=1)
rows = []
for aia_date, aia_path in aia_list:
    aia_dt = parse_date(aia_date)
    # Find closest HMI
    closest_hmi = None
    min_diff = timedelta.max
    for hmi_date, hmi_path in hmi_list:
        hmi_dt = parse_date(hmi_date)
        diff = abs(aia_dt - hmi_dt)
        if diff < min_diff:
            min_diff = diff
            closest_hmi = (hmi_date, hmi_path)
    if min_diff <= time_tolerance:
        # Read DATE-OBS from AIA FITS header using astropy
        try:
            with fits.open(aia_path) as hdul:
                # hdul.info()
                # for i, hdu in enumerate(hdul):
                #     print(f"Header {i}:")
                #     print(hdu.header)
                date_obs = hdul[1].header.get('DATE-OBS', '')
        except Exception as e:
            print(f"Error reading DATE-OBS from {aia_path}: {e}")
            date_obs = ''
        # Store relative paths from helio-torch folder
        rel_hmi_path = os.path.relpath(closest_hmi[1], start=os.path.dirname(os.path.dirname(__file__)))
        rel_aia_path = os.path.relpath(aia_path, start=os.path.dirname(os.path.dirname(__file__)))
        rows.append([aia_date, date_obs, rel_hmi_path, rel_aia_path])

# Write to CSV
csv_path = os.path.join(os.path.dirname(__file__), '../solar_data/data_index.csv')
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename_date', 'fits_DATE-OBS', 'hmi_path', 'aia_path'])
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {csv_path}")
