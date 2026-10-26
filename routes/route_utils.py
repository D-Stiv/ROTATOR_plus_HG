import pickle
import time
from tqdm import tqdm
from functools import lru_cache
import numpy as np
from heapq import heappop, heappush
from itertools import product


average_speeds = {
    "walk": 4*1000/3600,    # m/s
    "bike": 14*1000/3600,   # m/s
    "bus": 15*1000/3600,    # m/s
    "tram": 33*1000/3600,   # m/s
    "train": 45*1000/3600,  # m/s
    "drive": 27*1000/3600,  # m/s
    "transfer": 4*1000/3600,          # m/s for computation stability only
}



def _short_mode_segments(path_edges, min_mode_distance):
    """Find consecutive same-mode, non-transfer segments that are shorter than the configured minimum distance."""
    """
    Returns consecutive non-transfer same-mode segments whose total distance
    is shorter than min_mode_distance.

    Each segment is returned as a list of edge tuples:
      [((u, v), attr), ...]
    """
    if min_mode_distance is None or min_mode_distance <= 0:
        return []

    short_segments = []
    current_mode = None
    current_edges = []
    current_distance = 0.0

    def flush():
        """Store the current mode segment when it is a valid short segment."""
        if (current_mode is not None
            and current_mode != "transfer"
            and current_edges
            and current_distance < min_mode_distance):

            short_segments.append(list(current_edges))

    for edge in path_edges:
        (_, _), attr = edge
        mode = attr.get("mode")
        dist = attr.get("distance", 0.0)

        if mode == "transfer":
            flush()
            current_mode = None
            current_edges = []
            current_distance = 0.0
            continue

        if mode != current_mode:
            flush()
            current_mode = mode
            current_edges = [edge]
            current_distance = dist
        else:
            current_edges.append(edge)
            current_distance += dist

    flush()
    return short_segments

def _respects_min_mode_distance(path_edges, min_mode_distance):
    """Check whether a path satisfies the minimum distance required for same-mode segments."""
    return not _short_mode_segments(path_edges, min_mode_distance)


def _penalize_short_mode_segments(G, path_edges, min_mode_distance, penalty, weight_type, original_weights):
    """Increase the search cost of edges belonging to invalid short mode segments so alternative paths are favored."""
    """
    Multiplies weights for edges belonging to invalid short same-mode segments.
    """
    for segment in _short_mode_segments(path_edges, min_mode_distance):
        for (u, v), _ in segment:
            edge_key = (u, v)
            if edge_key not in original_weights:
                original_weights[edge_key] = G.edges[u, v].get(weight_type, 0.0)
            G.edges[u, v][weight_type] = original_weights[edge_key] * penalty

    return original_weights


def dijkstra_state_augmented(G, source_nodes, target_nodes, max_transfers=2, weight_type="distance"):
    """Find the lowest-cost path while tracking the number of transfers as part of the search state."""
    source_ids = {s[0] for s in source_nodes}
    target_ids = {t[0] for t in target_nodes}
    assert len(source_ids) == 1, "All source nodes must have the same node id"
    assert len(target_ids) == 1, "All target nodes must have the same node id"

    pq = []
    dist = {}
    parent = {}

    for s in source_nodes:
        dist[(s, 0)] = 0
        heappush(pq, (0, s, 0))

    best_target_state = None
    best_target_transfers = None

    while pq:
        cost, state, transfers = heappop(pq)

        if state in target_nodes:
            best_target_state = state
            best_target_transfers = transfers
            break

        for nbr in G[state]:
            edge = G[state][nbr]
            new_cost = cost + edge.get(weight_type, 0.0)
            new_transfers = transfers + (1 if edge.get("mode") == "transfer" else 0)

            if new_transfers > max_transfers:
                continue

            key = (nbr, new_transfers)
            if key not in dist or new_cost < dist[key]:
                dist[key] = new_cost
                parent[key] = ((state, transfers), ((state, nbr), edge))
                heappush(pq, (new_cost, nbr, new_transfers))

    if best_target_state is None:
        return None, float("inf")

    path = []
    cur = (best_target_state, best_target_transfers)

    while cur in parent:
        prev, edge = parent[cur]
        path.append(edge)
        cur = prev

    return list(reversed(path)), dist[(best_target_state, best_target_transfers)]

def _penalize_path_for_diversity(G, path_edges, penalty, weight_type, original_weights):
    """Penalize selected edges of a path to encourage subsequent path searches to produce more diverse routes."""
    if len(path_edges) < 2:
        deviation_indices = range(len(path_edges))
    else:
        deviation_indices = np.unique(np.round(np.logspace(0, np.log10(len(path_edges)), num=3)).astype(int) - 1)
        deviation_indices = [i for i in deviation_indices if 0 <= i < len(path_edges)]

    for idx in deviation_indices:
        (u, v), _ = path_edges[idx]
        edge_key = (u, v)
        if edge_key not in original_weights:
            original_weights[edge_key] = G.edges[u, v].get(weight_type, 0.0)
        try:
            G.edges[u, v][weight_type] *= penalty
        except (KeyError, TypeError):
            pass

    return original_weights

def verify_path_edges_in_graph(path_edges, G):
    """Verify that every edge in a path still exists in the graph."""
    for (u, v), _ in path_edges:
        if not G.has_edge(u, v):
            return False
    return True

def combine_pieces(
    paths_seq,
    G=None,
    max_distance=None,
    max_time=None,
    max_transfers=None,
    authorized_mode_sequences=None,
    unauthorized_mode_sequences=None,
    global_transfers=None,
    max_route_length=None,
    distance_thresh_minspeed=None,
    time_thresh_maxspeed=None,
    min_speed=None,
    max_speed=None,
    mode_distance_targets=None,
    mode_ratio_tolerance=0.15,
    transfer_ratio_policy="as_walk", # transfer edges are counted as walk distance for mode-ratio computation
):
    """Join pairwise route pieces into complete routes and apply the configured route constraints."""
    """
    Combine pairwise path pieces into full routes.

    Expected piece format:
        piece = (path_edges, weight, source_id, target_id)

    Expected path_edges format:
        path_edges = [((u, v), attr), ...]
        u = (node_id, mode)
        v = (node_id, mode)

    transfer_ratio_policy:
        - "exclude": transfer edges are excluded from mode-ratio denominator
        - "as_walk": transfer edges count as walk distance
        - "include": transfer edges are counted as transfer mode

    If return_route_stats is True, each returned route is:
        (full_path, total_weight, start_id, end_id, stats)

    Otherwise:
        (full_path, total_weight, start_id, end_id)
    """

    def get_piece_start_node(piece):
        """Return the first graph node of a route piece."""
        path_edges = piece[0]
        return path_edges[0][0][0] if path_edges else None

    def get_piece_end_node(piece):
        """Return the last graph node of a route piece."""
        path_edges = piece[0]
        return path_edges[-1][0][1] if path_edges else None

    def compute_target_ratios(targets):
        """Normalize requested mode-distance targets into relative mode ratios."""
        if not targets:
            return None

        clean_targets = {mode: value
            for mode, value in targets.items()
            if mode != "transfer" and value is not None}

        total = sum(clean_targets.values())

        if total <= 0:
            return None

        return {mode: value / total
            for mode, value in clean_targets.items()}

    def compute_mode_stats(path_edges, desired_ratios=None):
        """Calculate distance and distance-ratio statistics for each transport mode in a route."""
        mode_distances = {}

        if desired_ratios is not None:
            for mode in desired_ratios:
                mode_distances[mode] = 0.0

        for (_, _), attr in path_edges:
            mode = attr.get("mode")
            dist = attr.get("distance", 0.0)

            if mode == "transfer":
                if transfer_ratio_policy == "exclude":
                    continue
                if transfer_ratio_policy == "as_walk":
                    mode = "walk"
                elif transfer_ratio_policy == "include":
                    mode = "transfer"
                else:
                    raise ValueError("transfer_ratio_policy must be one of "
                        "{'exclude', 'as_walk', 'include'}"
                    )

            mode_distances[mode] = mode_distances.get(mode, 0.0) + dist

        denominator = sum(mode_distances.values())

        mode_ratios = {mode: (dist / denominator if denominator > 0 else 0.0)
            for mode, dist in mode_distances.items()}

        stats = {"mode_distances": mode_distances,
            "mode_ratios": mode_ratios,
            "ratio_denominator_distance": denominator,
            "transfer_policy": transfer_ratio_policy,}

        if desired_ratios is not None:
            stats["desired_mode_ratios"] = desired_ratios

        return stats

    def ratios_are_close(actual_ratios, desired_ratios):
        """Check whether actual mode ratios fall within the configured tolerance of the target ratios."""
        if desired_ratios is None:
            return True

        for mode, desired_ratio in desired_ratios.items():
            actual_ratio = actual_ratios.get(mode, 0.0)

            if isinstance(mode_ratio_tolerance, dict):
                tolerance = mode_ratio_tolerance.get(mode, 0.0)
            else:
                tolerance = mode_ratio_tolerance

            if abs(actual_ratio - desired_ratio) > tolerance:
                return False

        return True

    def route_conditions(path_edges):
        """Compute aggregate distance, time, effective time, mode sequence, and transfer statistics for a route."""
        conditions = {"total_distance": 0.0,
            "total_time": 0.0,
            "tot_eff_time": 0.0,
            "mode_sequence": [],
            "num_transfers": 0,
            "distance": {mode: 0.0 for mode in average_speeds.keys()},
            "time": {mode: 0.0 for mode in average_speeds.keys()},
            "eff_time": {mode: 0.0 for mode in average_speeds.keys()},}

        for (_, _), attr in path_edges:
            mode = attr.get("mode")
            dist = attr.get("distance", 0.0)

            avg_speed = average_speeds.get(mode, 0.0)
            time = (attr["time_weight"]
                if "time_weight" in attr
                else dist / avg_speed
                if avg_speed > 0
                else 0.0)
            eff_time = attr["time"] if "time" in attr else time

            conditions["total_distance"] += dist
            conditions["total_time"] += time
            conditions["tot_eff_time"] += eff_time

            if mode not in conditions["distance"]:
                conditions["distance"][mode] = 0.0
                conditions["time"][mode] = 0.0
                conditions["eff_time"][mode] = 0.0

            conditions["distance"][mode] += dist
            conditions["time"][mode] += time
            conditions["eff_time"][mode] += eff_time

            if mode != "transfer":
                if not conditions["mode_sequence"] or conditions["mode_sequence"][-1] != mode:
                    conditions["mode_sequence"].append(mode)

        conditions["num_transfers"] = max(len(conditions["mode_sequence"]) - 1, 0)
        return conditions

    def passes_route_filters(path_edges, stats):
        """Check whether a candidate route satisfies all configured length, speed, transfer, mode-sequence, and ratio constraints."""
        if max_route_length is not None and len(path_edges) > max_route_length:
            return False

        conditions = route_conditions(path_edges)

        if max_distance is not None and conditions["total_distance"] > max_distance:
            return False

        if max_time is not None and conditions["total_time"] > max_time:
            return False

        if global_transfers is not None and conditions["num_transfers"] > global_transfers:
            return False

        if max_transfers is not None and conditions["num_transfers"] > max_transfers:
            return False

        if distance_thresh_minspeed is not None:
            if conditions["total_distance"] > distance_thresh_minspeed:
                min_speed_value = (min_speed
                    if min_speed is not None
                    else 2 * average_speeds["walk"]
                )
                speed = (conditions["total_distance"] / conditions["tot_eff_time"]
                    if conditions["tot_eff_time"] > 0
                    else 0.0
                )
                if speed < min_speed_value:
                    return False

        if time_thresh_maxspeed is not None:
            if conditions["tot_eff_time"] > time_thresh_maxspeed:
                max_speed_value = (max_speed
                    if max_speed is not None
                    else average_speeds["tram"] / 2
                )
                speed = (conditions["total_distance"] / conditions["tot_eff_time"]
                    if conditions["tot_eff_time"] > 0
                    else 0.0
                )
                if speed > max_speed_value:
                    return False

        if unauthorized_mode_sequences is not None:
            for pattern in unauthorized_mode_sequences:
                pattern = tuple(pattern)
                plen = len(pattern)

                for i in range(len(conditions["mode_sequence"]) - plen + 1):
                    current = tuple(conditions["mode_sequence"][i:i + plen])
                    if current == pattern:
                        return False

        if authorized_mode_sequences is not None:
            authorized = {    tuple(pattern)
                for pattern in authorized_mode_sequences
            }

            if tuple(conditions["mode_sequence"]) not in authorized:
                return False

        if not ratios_are_close(stats["mode_ratios"], stats.get("desired_mode_ratios")):
            return False

        return True

    if not paths_seq:
        return []

    paths_seq = [piece_group for piece_group in paths_seq if piece_group]

    if not paths_seq:
        return []

    desired_ratios = compute_target_ratios(mode_distance_targets)

    full_routes = []


    max_len = max(len(piece_group) for piece_group in paths_seq)
    selected_combinations = []

    for i in range(max_len):
        combo = []

        for piece_group in paths_seq:
            selected_piece = None

            for offset in range(len(piece_group)):
                idx = (i + offset) % len(piece_group)
                candidate = piece_group[idx]

                first_node = get_piece_start_node(candidate)

                if not combo:
                    selected_piece = candidate
                    break

                prev_last_node = get_piece_end_node(combo[-1])

                if prev_last_node == first_node:
                    selected_piece = candidate
                    break

                if G is not None and G.has_edge(prev_last_node, first_node):
                    connector_attr = G.get_edge_data(prev_last_node, first_node)

                    connector_piece = ([((prev_last_node, first_node), connector_attr)],
                        connector_attr.get("weight", 0.0),
                        prev_last_node[0],
                        first_node[0],
                    )

                    combo.append(connector_piece)
                    selected_piece = candidate
                    break

            if selected_piece is None:
                combo = None
                break

            combo.append(selected_piece)

        if combo:
            selected_combinations.append(tuple(combo))

    all_combinations = selected_combinations

    for combo in all_combinations:
        full_path = []
        total_weight = 0.0
        valid_combo = True

        for piece in combo:
            path_edges, weight, _, _ = piece

            if not path_edges:
                valid_combo = False
                break

            first_node = get_piece_start_node(piece)

            if full_path:
                prev_last_node = full_path[-1][0][1]

                if prev_last_node != first_node:
                    if G is None or not G.has_edge(prev_last_node, first_node):
                        valid_combo = False
                        break

                    connector_attr = G.get_edge_data(prev_last_node, first_node)

                    full_path.append(((prev_last_node, first_node), connector_attr))
                    total_weight += connector_attr.get("weight", 0.0)

            full_path.extend(path_edges)
            total_weight += weight

        if not valid_combo:
            continue

        if G is not None and not verify_path_edges_in_graph(full_path, G):
            print("Invalid full route detected, skipping.")
            continue

        stats = compute_mode_stats(full_path, desired_ratios=desired_ratios)

        # discard extreme case: max(ratio)=1, max_desired ratio < 1-tolerance
        if 'desired_mode_ratios' in stats and max(stats['mode_ratios'].values()) > max(stats['desired_mode_ratios'].values()) + mode_ratio_tolerance:
            continue

        if not passes_route_filters(full_path, stats):
            continue

        route = (full_path,
            total_weight,
            combo[0][2],
            combo[-1][3],)

        full_routes.append(route)

    return full_routes


def k_feasible_paths(
    G,
    source_nodes,
    target_nodes,
    k=5,
    max_transfers=2,
    max_factor=2,
    penalty=4,
    overlap_threshold=0.5,
    weight_type="distance",
    min_mode_distance=None,
    max_attempts=None,
):
    """Generate up to k feasible, reasonably short, and diverse paths between source and target nodes."""
    def restore_original_weights(G, original_weights):
        """Restore graph edge weights that were temporarily modified during path generation."""
        for (u, v), weight in original_weights.items():
            G.edges[u, v][weight_type] = weight

    if max_attempts is None:
        max_attempts = max(k * 5, 10)

    original_weights = {}
    valid_paths = []
    seen_paths = set()

    source_id = source_nodes[0][0]
    target_id = target_nodes[0][0]

    best_path, best_cost = dijkstra_state_augmented(G,
        source_nodes,
        target_nodes,
        max_transfers=max_transfers,
        weight_type=weight_type,)

    if not best_path:
        return []

    min_cost = best_cost
    attempts = 0

    while len(valid_paths) < k and attempts < max_attempts:
        attempts += 1

        path_key = tuple(edge[0] for edge in best_path)

        if best_cost > max_factor * min_cost:
            break

        if path_key not in seen_paths:
            seen_paths.add(path_key)

            enforce_min_mode_distance = (min_mode_distance is not None
                and min_mode_distance > 0
                and any(attr.get("mode") == "transfer" for _, attr in best_path))

            respects_min_mode_distance = (not enforce_min_mode_distance
                or _respects_min_mode_distance(best_path, min_mode_distance))

            if respects_min_mode_distance:
                if verify_path_edges_in_graph(best_path, G):
                    valid_paths.append((best_path, best_cost, source_id, target_id))
            else:
                original_weights = _penalize_short_mode_segments(    G,
                    best_path,
                    min_mode_distance=min_mode_distance,
                    penalty=penalty,
                    weight_type=weight_type,
                    original_weights=original_weights,
                )

        # Penalize accepted paths too, to encourage diversity.
        original_weights = _penalize_path_for_diversity(G,
            best_path,
            penalty=penalty,
            weight_type=weight_type,
            original_weights=original_weights,)

        best_path, best_cost = dijkstra_state_augmented(G,
            source_nodes,
            target_nodes,
            max_transfers=max_transfers,
            weight_type=weight_type,)

        if not best_path:
            break

    if overlap_threshold < 1 and valid_paths:
        valid_paths, _, _ = refine_paths(l=valid_paths,
            num_paths=k,
            G=G,
            overlap_threshold=overlap_threshold,)

    restore_original_weights(G, original_weights)
    return valid_paths


def build_paths(
    G,
    matched_sequences,
    unique_dict,
    k=5,
    max_transfers=2,
    max_factor=10,
    penalty=4,
    mode_coefficients=None,
    max_distance=None,
    max_time=None,
    authorized_mode_sequences=None,
    unauthorized_mode_sequences=None,
    global_transfers=None,
    max_route_length=None,
    distance_thresh_minspeed=None,
    time_thresh_maxspeed=None,
    min_speed=None,
    max_speed=None,
    weight_type="distance",
    min_mode_distance=None,
    mode_distance_targets=None,
    mode_ratio_tolerance=0.15,
    transfer_ratio_policy="exclude",
    overlap_threshold=1,
):
    """Build route candidates for matched VPOI sequences by finding and combining feasible path pieces."""
    if weight_type == "distance":
        weight_type = "weight"
    else:
        weight_type = weight_type + "_weight"

    @lru_cache(maxsize=None)
    def cached_k_paths(u, v):
        """Cache feasible-path searches for a node pair to avoid repeating the same computation."""
        return tuple(k_feasible_paths(G,
                [u],
                [v],
                k=k,
                max_transfers=max_transfers,
                max_factor=max_factor,
                penalty=penalty,
                weight_type=weight_type,
                min_mode_distance=min_mode_distance,
                overlap_threshold=overlap_threshold,
                ))

    if not (mode_coefficients is None or mode_coefficients == {}):
        min_coeff = min(mode_coefficients.values())
        for _, _, data in tqdm(G.edges(data=True)):
            mode = data.get("mode")
            coeff = mode_coefficients.get(mode, min_coeff*0.5)  # default as smallest
            data[weight_type] = data.get(weight_type, 0.0) * coeff

    possible_sequences = []

    for seq in tqdm(matched_sequences, desc="Building training paths"):
        try:
            partial_sequences = []
            for u, v in zip(seq, seq[1:]):
                sources, targets = unique_dict[u], unique_dict[v]
                all_source_target = [list(st) for st in product(sources, targets)]

                if not partial_sequences:
                    partial_sequences = all_source_target
                else:
                    new_partial_sequences = []

                    for possible_seq in partial_sequences:
                        for st in all_source_target:
                            last_node = possible_seq[-1]
                            first_node = st[0]

                            if last_node[1] == first_node[1]:
                                new_partial_sequences.append(possible_seq + [st[1]])

                    partial_sequences, _ = get_valid_sequences(new_partial_sequences,
                        authorized_mode_sequences,
                        unauthorized_mode_sequences,
                    )
        except KeyError as e:
            print(f"[Check DB table of vpoi to understand why the corresponding node is missing in the graph]\nKeyError for sequence {seq}: {e}")
            continue

        possible_sequences.extend(partial_sequences)


    routes = []

    for seq in tqdm(possible_sequences, desc="Combining route pieces"):
        paths_seq = []
        complete = True

        for u, v in zip(seq, seq[1:]):
            paths = cached_k_paths(u, v)

            if paths:
                paths_seq.append(paths)
            else:
                break

        if not complete:
            continue

        seq_routes = combine_pieces(paths_seq,
            G,
            max_distance=max_distance,
            max_time=max_time,
            max_transfers=max_transfers,
            authorized_mode_sequences=authorized_mode_sequences,
            unauthorized_mode_sequences=unauthorized_mode_sequences,
            global_transfers=global_transfers,
            max_route_length=max_route_length,
            distance_thresh_minspeed=distance_thresh_minspeed,
            time_thresh_maxspeed=time_thresh_maxspeed,
            min_speed=min_speed,
            max_speed=max_speed,
            mode_distance_targets=mode_distance_targets,
            mode_ratio_tolerance=mode_ratio_tolerance,
            transfer_ratio_policy=transfer_ratio_policy,
            )

        if seq_routes:
            routes.extend(seq_routes)

    return routes, (len(routes), len(possible_sequences))


def build_routes(
    G,
    vpoi_nodes_df,
    foldername='name of the folder where to save the output routes',
    vpoi_sequences=None,
    k=50,
    max_transfers=5,
    penalties=2,
    max_factors=2,
    save=True,
    batch_size=50000,
    mode_coefficients=None,
    transport_modes=None,
    max_distance=None,
    max_time=None,
    authorized_mode_sequences=None,
    unauthorized_mode_sequences=None,
    global_transfers=None,
    max_route_length=None,
    distance_thresh_minspeed=None,
    time_thresh_maxspeed=None,
    min_speed=None,
    max_speed=None,
    weight_type='distance',
    overlap_threshold=1,
    min_mode_distance=None,
    mode_distance_targets=None,
    mode_ratio_tolerance=0.15,
    transfer_ratio_policy="exclude",
):
    """Prepare graph and VPOI data, generate training routes under multiple parameter settings, and optionally save them in batches."""
    """
    Build training routes from vpoi sequences and save them in parts.

    New parameters:
        - min_mode_distance : float | None
            Minimum distance required for each consecutive same-mode non-transfer segment.
            If None or <= 0, the constraint is disabled.

        - mode_distance_targets : dict[str, float] | None
            Desired relative distance composition by mode, e.g. {'walk': 1, 'bus': 3}.
            Values are normalized internally.

        - mode_ratio_tolerance : float | dict[str, float]
            Allowed absolute deviation from target mode ratios.

        - transfer_ratio_policy : str
            How transfer edges are handled in mode-ratio computation:
            'exclude', 'as_walk', or 'include'.

        - return_route_stats : bool
            If True, route tuples include a fifth element with mode-distance stats.
    """

    # Normalize scalar hyperparameters to lists.
    if not isinstance(max_transfers, (list, tuple, set)):
        max_transfers = [max_transfers]
    if not isinstance(penalties, (list, tuple, set)):
        penalties = [penalties]
    if not isinstance(max_factors, (list, tuple, set)):
        max_factors = [max_factors]

    # Filter out edges with modes not in transport_modes.
    if transport_modes is not None:
        print(f'Filtering edges to keep only transport modes: {transport_modes}')
        edges_to_remove = []

        for u, v, data in G.edges(data=True):
            if data.get('mode') not in transport_modes:
                edges_to_remove.append((u, v))

        G.remove_edges_from(edges_to_remove)
        print(f'Removed {len(edges_to_remove)} edges not in transport modes {transport_modes}.')

    save = save and (foldername is not None)

    target_col = 'close_node_id'
    source_col = 'poi_id'

    mapping_csv_graph_node = dict(zip(vpoi_nodes_df[source_col], vpoi_nodes_df[target_col]))

    mapped_sequences = [
        [mapping_csv_graph_node[n] for n in seq if n in mapping_csv_graph_node]
        for seq in vpoi_sequences
    ]

    print(f"Number of vpoi sequences: {len(vpoi_sequences)}")

    matched_sequences = []
    not_matched_idx = []

    for idx, seq in enumerate(mapped_sequences):
        if len(seq) <= 1:
            not_matched_idx.append(idx)
            continue
        matched_sequences.append(seq)

    all_ids_in_sequences = {n for seq in matched_sequences for n in seq}
    unique_ids = {n[0] for n in G.nodes}
    vpoi_ids = set(vpoi_nodes_df['close_node_id'].tolist())
    vpoi_in_unique = vpoi_ids.intersection(unique_ids)

    training_ids = all_ids_in_sequences.intersection(vpoi_in_unique)

    print(f"Number of matched sequences: {len(matched_sequences)}")

    unique_dict = {uid: [] for uid in training_ids}

    for n in G.nodes:
        if n[0] in unique_dict and (transport_modes is None or n[1] in transport_modes):
            unique_dict[n[0]].append(n)

    list_matched_sequences = [matched_sequences]

    global_routes = []
    partial_routes = []

    count = 0
    t0 = t_start = time.time()
    perc = {'num_eff_routes': 0, 'num_possible_routes': 0}

    for mt in tqdm(max_transfers, "Max transfers"):
        for penalty in penalties:
            for max_factor in max_factors:
                for _matched_sequences in tqdm(list_matched_sequences, "Matched sequences"):
                    g_routes, stats = build_paths(G,
                        matched_sequences=_matched_sequences,
                        unique_dict=unique_dict,
                        k=k,
                        max_transfers=mt,
                        max_factor=max_factor,
                        penalty=penalty,
                        mode_coefficients=mode_coefficients,
                        max_distance=max_distance,
                        max_time=max_time,
                        authorized_mode_sequences=authorized_mode_sequences,
                        unauthorized_mode_sequences=unauthorized_mode_sequences,
                        global_transfers=global_transfers,
                        max_route_length=max_route_length,
                        distance_thresh_minspeed=distance_thresh_minspeed,
                        time_thresh_maxspeed=time_thresh_maxspeed,
                        min_speed=min_speed,
                        max_speed=max_speed,
                        weight_type=weight_type,
                        overlap_threshold=overlap_threshold,
                        min_mode_distance=min_mode_distance,
                        mode_distance_targets=mode_distance_targets,
                        mode_ratio_tolerance=mode_ratio_tolerance,
                        transfer_ratio_policy=transfer_ratio_policy,
                    )

                    perc['num_eff_routes'] += stats[0]
                    perc['num_possible_routes'] += stats[1]

                    global_routes.extend(g_routes)
                    partial_routes.extend(g_routes)

                    should_save_batch = (save
                        and len(partial_routes) > 0
                        and (count == 0 or len(partial_routes) // batch_size >= 1)
                    )

                    if should_save_batch:
                        print(f'Saving partial routes with {len(partial_routes)} entries... '
                            f'time elapsed: {time.time() - t0:.1f}s'
                        )

                        # Save partial routes to a file with a count suffix.

                        # save_partial(foldername,
                        #     partial_routes,
                        #     G,
                        #     vpoi_ids=training_ids,
                        #     count=f'{count:03d}',
                        # )

                        count += 1
                        partial_routes = []
                        t0 = time.time()

    if partial_routes and save:
        print(f'Saving final partial routes with {len(partial_routes)} entries... '
            f'time elapsed: {time.time() - t0:.1f}s')

        # save_partial(foldername,
        #     partial_routes,
        #     G,
        #     vpoi_ids=training_ids,
        #     count=f'{count:03d}',)

        partial_routes = []

    if save and len(global_routes) > 0:
        # dt = datetime.now().strftime("%Y%m%d_%H%M%S") if debug else None

        print(f'Saving all routes with {len(global_routes)} entries... '
            f'total time elapsed: {time.time() - t_start:.1f}s')

        # save_partial(foldername,
        #     global_routes,
        #     G,
        #     vpoi_ids=training_ids,
        #     count=dt,)

        print('All done.')

    return global_routes



def build_routes_from_sequences(
    vpoi_sequences, G, mapping_osm_graph,
    k=50, max_transfers=[3], penalties=[4], max_factors=[2],
    mode_coefficients=None, transport_modes=None, max_distance=None,
    max_time=None, authorized_mode_sequences=None, unauthorized_mode_sequences=None,
    global_transfers=None, max_route_length=None, distance_thresh_minspeed=None,
    time_thresh_maxspeed=None, min_speed=None, max_speed=None, list_weights=None,
    overlap_threshold=1, min_mode_distance=None, 
    mode_distance_targets=None, mode_ratio_tolerance=0.15,
    file_path='path where to save the output routes',
):

    """Generate routes for each VPOI sequence and each requested edge-weight type, then save the resulting routes."""
    count = 0

    for s_idx, seq in enumerate(vpoi_sequences):
        print(f'Sequence {s_idx}: {seq}')

        for weight_type in list_weights:
            print(f'Building routes for weight type: {weight_type} ...')

            output_routes = build_routes(G=G,
                vpoi_nodes_df=mapping_osm_graph,
                vpoi_sequences=[seq],
                k=k,
                max_transfers=max_transfers,
                penalties=penalties,
                max_factors=max_factors,
                mode_coefficients=mode_coefficients,
                transport_modes=transport_modes,
                max_distance=max_distance,
                max_time=max_time,
                authorized_mode_sequences=authorized_mode_sequences,
                unauthorized_mode_sequences=unauthorized_mode_sequences,
                global_transfers=global_transfers,
                max_route_length=max_route_length,
                distance_thresh_minspeed=distance_thresh_minspeed,
                time_thresh_maxspeed=time_thresh_maxspeed,
                min_speed=min_speed,
                max_speed=max_speed,
                weight_type=weight_type,
                overlap_threshold=overlap_threshold,
                min_mode_distance=min_mode_distance,
                mode_distance_targets=mode_distance_targets,
                mode_ratio_tolerance=mode_ratio_tolerance,)
            
            
            with open(file_path, "wb") as f:
                pickle.dump(output_routes, f)

            print(f'Sample routes saved to {file_path}')





def refine_paths(l, num_paths, G, overlap_threshold=0.3, resolution=10):
    """Select a diverse subset of candidate paths based on their overlap in H3 hexagons."""
    """
    Select diverse paths based on H3 hexagon overlap.

    Args:
        l:
            List of route tuples:
                (path_edges, total_weight, source_id, target_id)
            or:
                (path_edges, total_weight, source_id, target_id, stats)

        num_paths:
            Desired number of diverse paths.

        G:
            NetworkX graph.

        overlap_threshold:
            Maximum allowed overlap between paths.

        resolution:
            H3 hexagon resolution.

    Returns:
        f:
            Selected diverse paths.

        hexagons_list:
            List of hexagon sets for each path in f.

        selected_perc:
            Overlap percentages associated with selected paths.
    """
    if not l:
        return [], [], []

    hex_cache = {}

    all_hexagons = [
        path_to_hexagons(path, G, resolution=resolution, hex_cache=hex_cache)
        for path in l
    ]

    f = [l[0]]
    hexagons_list = [all_hexagons[0]]
    selected_perc = [np.float64(0.0)]

    remaining_indices = set(range(1, len(l)))

    while len(f) < num_paths and remaining_indices:
        best_candidate = None
        best_overlap = float("inf")
        best_index = -1

        for i in list(remaining_indices):
            candidate_hex = all_hexagons[i]

            if len(candidate_hex) == 0:
                continue

            overlap_with_selected = [
                len(candidate_hex & selected_hex) / len(candidate_hex)
                for selected_hex in hexagons_list
            ]

            if any(overlap > overlap_threshold for overlap in overlap_with_selected):
                continue

            overlap_with_remaining = []

            for j in remaining_indices - {i}:
                other_hex = all_hexagons[j]

                if len(other_hex) == 0:
                    continue

                overlap = len(candidate_hex & other_hex) / len(candidate_hex)

                if overlap > overlap_threshold:
                    overlap_with_remaining = None
                    break

                overlap_with_remaining.append(overlap)

            if overlap_with_remaining is None:
                continue

            current_min = (
                np.min(overlap_with_remaining)
                if overlap_with_remaining
                else 0.0
            )

            if current_min < best_overlap:
                best_overlap = current_min
                best_candidate = l[i]
                best_index = i

        if best_candidate is None:
            break

        f.append(best_candidate)
        hexagons_list.append(all_hexagons[best_index])
        selected_perc.append(best_overlap)
        remaining_indices.remove(best_index)

    return f, hexagons_list, selected_perc


from h3.api import basic_str as h3
def path_to_hexagons(path, G, resolution=10, hex_cache={}):
    """Convert the geographic edges of a path into the set of H3 cells touched by the path."""
    """Convert a path to a set of H3 hexagons covering its edges."""
    path_key = str(path)
    if path_key in hex_cache:
        return hex_cache[path_key]

    hexagons = set()
    for e1, e2 in zip(path[0][:-1], path[0][1:]):
        # Get edge data
        u = e1[0]
        v = e2[0]
        edge_data = G[u][v]
        
        # Get coordinates - assuming each node has 'x' and 'y' attributes (lon/lat)
        u_coords = (G.nodes[u]['lon'], G.nodes[u]['lat'])
        v_coords = (G.nodes[v]['lon'], G.nodes[v]['lat'])
        
        # Hexagonize endpoints
        hexagons.add(h3.latlng_to_cell(u_coords[1], u_coords[0], resolution))
        hexagons.add(h3.latlng_to_cell(v_coords[1], v_coords[0], resolution))
        
        # Optionally: Add hexagons along the line if edge geometry is available
        if 'geometry' in edge_data:
            coords = list(edge_data['geometry'].coords)
            for lon, lat in coords:
                hexagons.add(h3.latlng_to_cell(lat, lon, resolution))
    # Cache the hexagons for this path
    hex_cache[path_key] = hexagons
    return hexagons


def get_valid_sequences(partial_sequences, authorized_mode_sequences=None, unauthorized_mode_sequences=None):
    """Filter candidate node sequences according to authorized and unauthorized transport-mode patterns."""
    if unauthorized_mode_sequences is None and authorized_mode_sequences is None:
        return partial_sequences, None
    if unauthorized_mode_sequences is not None:
        # add [drive, walk, drive], [walk, drive, walk] because we are sure sart/end nodes are different
        # add [walk, drive], [drive, walk] because we know that start to end location are generally far from each other. Probably mot building to parking spot or vice-versa
        unauthorized_mode_sequences.extend([['drive', 'walk', 'drive'], ['walk', 'drive', 'walk'], ['walk', 'drive'], ['drive', 'walk']])
    
    valid_sequences = []
    mode_seqs = []
    for seq in partial_sequences:
        mode_sequence = [node[1] for node in seq]
        mode_sequence = list(np.unique(mode_sequence))  # remove consecutive duplicates
        if unauthorized_mode_sequences is not None:
            mismatch = False
            for pattern in unauthorized_mode_sequences:
                plen = len(pattern)
                for i in range(len(mode_sequence) - plen + 1):
                    if tuple(mode_sequence[i:i+plen]) == tuple(pattern):
                        mismatch = True
                        break
            if mismatch:
                continue
        if authorized_mode_sequences is not None:
            match = False
            for pattern in authorized_mode_sequences:
                if tuple(mode_sequence) == tuple(pattern):
                    match = True
                    break
            if not match:
                continue
        valid_sequences.append(seq)
        mode_seqs.append(mode_sequence)
    return valid_sequences, mode_seqs
