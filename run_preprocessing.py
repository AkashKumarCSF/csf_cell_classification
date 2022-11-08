"""Cropping of patches from liquor ROIs"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from lxml import etree
from scipy.spatial.distance import cdist
from tqdm import tqdm


def find_files(path):
    """ Recursively find all matching .png and .roi in path. """
    result = []
    for root, _, files in os.walk(path):
        rois = [f for f in files if f.endswith('.roi')]
        pngs = [f for f in files if f.endswith('.png')]
        for png in pngs:
            roi = png + '.roi'
            if not roi in rois:
                print(
                    f'WARNING: No associated .roi file for {os.path.join(root, png)}')
        for roi in rois:
            png = roi[:-4]
            if not png in pngs:
                print(
                    f'WARNING: No associated .png file for {os.path.join(root, roi)}')
                continue
            result.append((root, png, roi))
    return result


def get_case_mapping(group_by_cases):
    """ Build mapping liquor -> case (patient) from spreadsheet. """
    df = pd.read_excel(group_by_cases,
                       engine='odf')  # engine odf for openoffice
    df = df.dropna(
        how='all')  # odf engine adds rows of all NaNs to end of table
    df = df.fillna(method='ffill')  # fill merged cells in table
    liquor_to_case = {liquor: int(df[df['Index'] == liquor]['Case'].item()) for
                      liquor in df['Index']}
    return liquor_to_case


def fix_annotation(annotation):
    """ Correct some sample annotations with inconsistent spelling. """
    if annotation == 'Aktivierter Monozyt':
        annotation = 'aktivierter Monozyt'
    elif annotation == 'Hämatoidin(Kristall)':
        annotation = 'Hämatoidin'
    # Hee: Hallo, bitte bei den Zelltypen 'amitotische Plasmazelle' und 'Plasmazelle'
    # zu letzterem vereinen, als eine Kategorie.
    elif annotation == 'amitotische Plasmazelle':
        annotation = 'Plasmazelle'
    return annotation


def extract_sample_points(img, sample, cropsize, padding_size):
    # Extract sample points
    features = sample.findall('Feature')
    points = np.zeros((len(features) // 2, 2), dtype=int)
    for feature in features:
        name = feature.get('Name')
        dim = int(name[0] == 'y')
        idx = int(name[1:])
        val = int(feature.get('Value'))
        points[idx][dim] = val + padding_size

    # Determine points where crops are inside borders of the UNPADDED image
    height, width = img.shape[:2]
    lower_left = np.array([cropsize // 2 + padding_size, cropsize // 2 + padding_size])
    upper_right = np.array([width - (cropsize // 2) - padding_size,
                            height - (cropsize // 2) - padding_size])
    inside_borders = np.all(np.logical_and(lower_left <= points, points <= upper_right), axis=1)

    if padding_size == 0:
        # No padding used, so drop points where crops exceed image borders
        points = points[inside_borders]
        inside_borders = inside_borders[inside_borders]
    else:
        # Sanity check: all crops should lie inside borders of the PADDED image
        lower_left = np.array([cropsize // 2, cropsize // 2])
        upper_right = np.array([width - (cropsize // 2),
                                height - (cropsize // 2)])
        assert np.all(np.logical_and(lower_left <= points, points <= upper_right))

    return points, inside_borders


def max_points_distance(points):
    """ Calculate max pairwise distance if there are multiple points for one sample. """
    if points.shape[0] > 1:
        distance_matrix = cdist(points, points, metric='chebyshev')
        triu_indices = np.triu_indices_from(distance_matrix, k=1)
        max_distance = distance_matrix[triu_indices].max()
    else:
        max_distance = 0
    return max_distance


def save_crop(dest: Path, img, points, cropsize: int):
    # If there are multiple points for a sample, we take only one crop around the first point
    # Note that we read points in standard (x,y) format, but opencv uses (y,x)
    crop = img[points[0, 1] - (cropsize // 2): points[0, 1] + (cropsize // 2),
           points[0, 0] - (cropsize // 2): points[0, 0] + (cropsize // 2)]
    assert crop.shape == (cropsize, cropsize, 3)
    assert not dest.exists()
    cv2.imwrite(str(dest), crop)


def main(args):
    files = find_files(args.datapath)
    if args.group_by_cases is None:
        stats = pd.DataFrame()
    else:
        liquor_to_case = get_case_mapping(args.group_by_cases)
        stats = pd.DataFrame(
            index=sorted(set(liquor_to_case.values()))).rename_axis('Case',
                                                                    axis=0)
    log = pd.DataFrame(
        columns=['src', 'dest', 'class', 'x', 'y', 'inside_borders'])

    out_path = str(Path(args.statsfile).parent)
    dry_run = args.dry_run
    if dry_run:
        print(f"Would save to {out_path}")
    else:
        os.makedirs(out_path, exist_ok=True)

    for root, png_file, roi_file in tqdm(files):
        roi = etree.parse(os.path.join(root, roi_file))
        img = cv2.imread(os.path.join(root, png_file))
        if args.group_by_cases is None:
            liquor = Path(png_file).stem
            case = Path(root).stem
        else:
            liquor = '_'.join(png_file.split('_')[:2])
            case = liquor_to_case[liquor]

        samples = roi.findall('Sample')

        if dry_run:
            tqdm.write(
                f"Would process case {case}, liquor {liquor} with {len(samples)} samples.")
            continue

        if args.padding:
            padding_size = args.cropsize // 2 + 1
            img = cv2.copyMakeBorder(img, *([padding_size] * 4), cv2.BORDER_REFLECT_101)
        else:
            padding_size = 0

        for idx, sample in enumerate(samples):
            sample_class = fix_annotation(sample.get('Class'))
            points, inside_borders = extract_sample_points(img, sample,
                                                           args.cropsize,
                                                           padding_size)

            if points.size == 0:
                continue

            crop_dir = Path(out_path) / sample_class
            crop_dir.mkdir(exist_ok=True)
            crop_dest = crop_dir / f'{case}_{Path(png_file).stem}-{idx}.png'
            save_crop(crop_dest, img, points, args.cropsize)

            if sample_class not in stats:
                stats[sample_class] = 0
            if case not in stats.index:
                stats.loc[case] = 0
            stats.loc[case][sample_class] += 1

            log = log.append({'src': os.path.join(root, png_file),
                              'dest': crop_dest, 'class': sample_class,
                              'x': points[0, 0] - padding_size,
                              'y': points[0, 1] - padding_size,
                              'inside_borders': inside_borders[0]},
                             ignore_index=True)

    stats.loc['total'] = stats.sum(axis=0)
    stats['total'] = stats.sum(axis=1)
    print(stats.to_markdown())

    if not dry_run:
        statsfile
        stats.to_csv(args.statsfile)
        log.to_csv(os.path.join(out_path, 'preprocessing_log.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--datapath',
                        help='Traverses all subdirectories to find .png and .roi files' + \
                             'Matching .png and .roi files must be in same directory.')
    parser.add_argument('-o', '--statsfile',
                        default=None)
    parser.add_argument('-g', '--group-by-cases',
                        default=None,
                        help='Spreadsheet file associating liquors with cases (patients).')
    parser.add_argument('-s', '--cropsize', type=int, default=224)
    parser.add_argument('-n', '--dry_run', dest='dry_run', action='store_true')
    parser.add_argument('-r', '--padding', dest='padding', action='store_true',
                        help='Use border reflect padding such that crops can be taken' + \
                             'that would otherwise exceed image borders.')
    args = parser.parse_args()
    main(args)
