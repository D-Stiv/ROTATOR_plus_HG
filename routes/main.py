import argparse
import numpy as np
from tqdm import tqdm
import pickle
import pandas as pd


transport_modes = {
    "wangerland": ["walk", "bike", "drive", "bus", "transfer"], 
}

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--penalty", type=float, default=4, help="Penalty for diversity")
    parser.add_argument("--max_transfers", type=int, default=2, help="Maximum number of transfers allowed in the route")
    parser.add_argument("--max_distance", type=float, default=None, help="Maximum total distance of the route (in meters)")
    parser.add_argument("--max_time", type=float, default=None, help="Maximum total time of the route (in seconds)")
    parser.add_argument("--authorized_mode_sequences", type=str, default=None, help="authorized mode sequences (e.g., ['walk', 'bus', 'walk'])", nargs='+')
    parser.add_argument("--unauthorized_mode_sequences", type=str, default=None, help="unauthorized mode sequences (e.g., ['drive', 'walk'])", nargs='+')    
    parser.add_argument("--global_transfers", type=int, default=None, help="Maximum total number of transfers allowed in the route")
    parser.add_argument("--k", type=int, default=50, help="number of alternatives in shortest path search")
    parser.add_argument("--mode_coefficients", type=str, default="[{'transfer': 2, 'walk': 1, 'bike': 0.5, 'drive': 0.1, 'bus': 0.1, 'train': 0.1, 'tram': 0.1}]", help="Mode coefficients (e.g., walk:1.0, bike:2.0)")

    parser.add_argument("--distance_thresh_minspeed_m", type=float, default=None, help="Distance threshold in meters for applying minimum speed filter (e.g., 1000 for 1 km)")
    parser.add_argument("--time_thresh_maxspeed_s", type=float, default=None, help="Time threshold in seconds for applying maximum speed filter (e.g., 1800 for 30 minutes)")
    parser.add_argument("--min_speed_mps", type=float, default=None, help="Minimum speed in meters per second (e.g., 0.5 for 1.8 km/h). If None, we take 2X walk speed)")
    parser.add_argument("--max_speed_mps", type=float, default=None, help="Maximum speed in meters per second (e.g., 10 for 36 km/h). If None, we take tram speed / 2)")
    parser.add_argument("--overlap_threshold", type=float, default=1.0, help="Threshold for overlapping hexagons. 1 --> full oveerlap allowed, 0 --> no overlap allowed", nargs='+')
    parser.add_argument("--list_weights", type=str, default=None, help="List of weights for each edge. e,g: time, distance")

    # complex parameters
    parser.add_argument("--min_dist_mode", type=float, default=None, help="Minimum distance in meters for consecutive edges with the same mode (e.g., 500 for 500 meters). If None, no minimum distance constraint is applied.")
    parser.add_argument("--modes_prop", type=str, default=None, help="Proportion of modes (distance ratio) to consider (e.g., [{'walk': 0.5, 'bike': 0.3, 'drive': 0.2}]). If None, all modes are considered and no specific proportion is applied.")
    parser.add_argument("--m_prop_tolerance", type=float, default=0.05, help="Tolerance for mode proportions (distance ratio) to consider (e.g., {'walk': 0.1, 'bike': 0.1, 'drive': 0.1}). If None, no tolerance is applied.")
    parser.add_argument("--m_prop_with_transfer", action='store_true', help="If true, include transfer mode as walk, otherwise exclude transfer mode from the modes_prop calculation")

    args = parser.parse_args()
    return args

from routes.route_utils import build_routes_from_sequences

if __name__ == "__main__":
    args = parse_arguments()

    k = args.k
    max_transfers = args.max_transfers
    global_transfers = args.global_transfers
    penalties = args.penalties
    max_factors = args.max_factors

    authorized_mode_sequences = [eval(seq) for seq in args.authorized_mode_sequences] if args.authorized_mode_sequences else None
    unauthorized_mode_sequences = [eval(seq) for seq in args.unauthorized_mode_sequences] if args.unauthorized_mode_sequences else None
    mode_coefficients = [eval(mc) for mc in args.mode_coefficients] if args.mode_coefficients else None
    modes_props = eval(args.modes_prop) if args.modes_prop else None
    list_weights = eval(args.list_weights) if args.list_weights else None



    # load graph
    graph_path_ = f"./resources/graph.pkl"
    with open(graph_path_, 'rb') as f:
        G = pickle.load(f)
    print(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges.")

    # load POI sequences samples
    poi_sequences_path_ = f"./resources/poi_seq_samples.csv"
    poi_sequences_df = pd.read_csv(poi_sequences_path_)

    # mapping POI id to graph ID
    mapping_path = f"./resources/mapping_poi_id_graph.csv"
    mapping_df = pd.read_csv(mapping_path)


    for mode_distance_target in tqdm(modes_props, "Start: modes proportion"):
        transport_modes = ['transfer'] + [mode for mode in mode_distance_target.keys() if mode_distance_target[mode] > 0]
        output = build_routes_from_sequences(vpoi_sequences=poi_sequences_df, 
                                            G=G, mapping_osm_graph=mapping_df,
                                            k=k, penalties=penalties, max_factors=max_factors, max_transfers=max_transfers,
                                            mode_coefficients=mode_coefficients, transport_modes=transport_modes, 
                                            max_distance=args.max_distance, max_time=args.max_time, authorized_mode_sequences=authorized_mode_sequences, unauthorized_mode_sequences=unauthorized_mode_sequences,
                                            global_transfers=args.global_transfers,
                                            distance_thresh_minspeed=args.distance_thresh_minspeed_m, time_thresh_maxspeed=args.time_thresh_maxspeed_s, min_speed=args.min_speed_mps, max_speed=args.max_speed_mps, 
                                            list_weights=list_weights,
                                            overlap_threshold=args.overlap_threshold,
                                            min_mode_distance=args.min_dist_mode, mode_distance_targets=mode_distance_target,
                                            mode_ratio_tolerance = args.m_prop_tolerance if args.modes_prop else None,
                                )  





