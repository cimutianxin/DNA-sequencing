import config
import sys
import pyabf
import level_finder as lf
import level_category as lc
from config import seq_A, seq_B, ref_seqs_map

if __name__ == "__main__":
    abf = pyabf.ABF(config.get_data_path(sys.argv[1]))
    abf_name = sys.argv[1]

    abf = lf.downsample(abf, 100)
    i_ranges, t_ranges = lf.find_events(abf)

    abf = lf.extra_process(abf)
    change_points = lf.change_points(abf, i_ranges)

    lvl = lf.calc_level(abf, change_points[0])

    # base_lvl_map = lc.classify_level(change_points, seq_A)
