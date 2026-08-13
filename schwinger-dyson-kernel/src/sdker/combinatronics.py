from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# One pairing configuration on [1..n]:
#   pairs     = tuple of (a,b) with a < b
#   unmatched = tuple of indices in [1..n] that are not in any pair
Config = Tuple[Tuple[Tuple[int, int], ...], Tuple[int, ...]]


_pairing_cache: Dict[Tuple[int, int], "NonCrossingPairingCache"] = {}

@dataclass(frozen=True)
class NonCrossingPairingCache:
    """
    Immutable cache of saturated, non-crossing partial matchings on [1..N],
    together with (d,n) lookup tables.

    Parameters
    ----------
    N : int
        Maximum n for which levels and tables are built.

    max_d : int
        Maximum d for which (d,n) tables are built.

    levels[n] : tuple of Config
        All saturated, non-crossing configurations on [1..n] (no d-constraint).

    first_close[n] : tuple of int
        For each config in levels[n], earliest right endpoint b among its pairs
        (a,b). If the config has no pairs, this is N+1.

    table_dn[n][d] : tuple of Config
        All configs on [1..n] such that no pair (a,b) satisfies a <= d and
        b <= d. For n >= 1 and d > n, this is a single trivial config with
        no pairs and all indices unmatched.
    """
    N: int
    max_d: int
    levels: Tuple[Tuple[Config, ...], ...]
    first_close: Tuple[Tuple[int, ...], ...]
    table_dn: Tuple[Tuple[Tuple[Config, ...], ...], ...]


################################################
# Internal builders 
################################################

def _build_saturated_levels(N: int) -> Tuple[List[List[Config]], List[List[int]]]:
    """
    Internal: build all saturated, non-crossing configurations on [1..n]
    for n = 0..N using the 'last unmatched' recursion (see proof of Prop 9 in paper).

    Returns
    -------
    levels : List[List[Config]]
        levels[n] is the list of Config on [1..n].

    first_close : List[List[int]]
        first_close[n][k] is the smallest right endpoint b in any pair (a,b)
        in levels[n][k], or N+1 if there are no pairs.
    """
    if N < 0:
        raise ValueError(f"N must be >= 0, got {N}")

    levels: List[List[Config]] = [[] for _ in range(N + 1)]
    first_close: List[List[int]] = [[] for _ in range(N + 1)]

    # n = 0: empty interval [1..0]
    levels[0] = [((), ())]
    first_close[0] = [N+1] # no closed pairs yet

    # n = 1..N
    for n in range(1, N + 1):
        prev_cfgs = levels[n - 1]
        prev_fc = first_close[n - 1]

        cur_cfgs: List[Config] = []
        cur_fc: List[int] = []

        for idx, (pairs, unmatched) in enumerate(prev_cfgs):
            fc = prev_fc[idx]

            # Option A: n is unmatched
            cfgA: Config = (pairs, unmatched + (n,))
            cur_cfgs.append(cfgA)
            cur_fc.append(fc)  # earliest closing unchanged

            # Option B: pair n with last unmatched
            if unmatched:
                k = unmatched[-1]
                new_pairs = pairs + ((k, n),)
                new_unmatched = unmatched[:-1]

                # If we already had a closing <= n-1, keep it;
                # otherwise this is the first pair, closing at n.
                new_fc = fc if fc <= n - 1 else n

                cfgB: Config = (new_pairs, new_unmatched)
                cur_cfgs.append(cfgB)
                cur_fc.append(new_fc)

        levels[n] = cur_cfgs
        first_close[n] = cur_fc

    return levels, first_close


def _build_dn_table(
    N: int,
    max_d: int,
    levels: List[List[Config]],
    first_close: List[List[int]],
    restrict_trivial: bool = True,
) -> List[List[List[Config]]]:
    """
    Internal: build table_dn[n][d] from levels and first_close.

    table_dn[n][d] = list of configs on [1..n] such that no pair (a,b)
    has a <= d and b <= d. If restrict_trivial is True, then for n >= 1
    and d > n we enforce a single trivial config with all indices
    unmatched.
    """
    if max_d < 0:
        raise ValueError(f"max_d must be >= 0, got {max_d}")

    table_dn: List[List[List[Config]]] = [
        [ [] for _ in range(max_d + 1) ]
        for _ in range(N + 1)
    ]

    # n = 0: empty interval, single config for all d
    for d in range(max_d + 1):
        table_dn[0][d] = [levels[0][0]]

    # n >= 1
    for n in range(1, N + 1):
        cfgs_n = levels[n]
        fcs_n = first_close[n]
        d_cap = min(max_d, n)

        # Fill d <= n using first_close (earliest right endpoint)
        for cfg_idx, cfg in enumerate(cfgs_n):
            b_min = fcs_n[cfg_idx]  # N+1 if no pairs
            # Config allowed for all d with 0 <= d < b_min,
            # but we also cap by d_cap.
            max_allowed_d = min(b_min - 1, d_cap)
            if max_allowed_d >= 0:
                for d in range(0, max_allowed_d + 1):
                    table_dn[n][d].append(cfg)

        # For d > n: either trivial-only or copy from d_cap
        if restrict_trivial:
            trivial_cfg: Config = ((), tuple(range(1, n + 1)))
            for d in range(n + 1, max_d + 1):
                table_dn[n][d] = [trivial_cfg]
        else:
            for d in range(n + 1, max_d + 1):
                table_dn[n][d] = list(table_dn[n][d_cap])

    return table_dn


def _build_cache(N: int, max_d: int) -> NonCrossingPairingCache:
    """
    Internal: build a new NonCrossingPairingCache for (N, max_d).
    """
    # 1) Base DP for all levels
    levels_list, first_close_list = _build_saturated_levels(N)

    # 2) (d,n) tables
    table_dn_list = _build_dn_table(N, max_d, levels_list, first_close_list)

    # 3) Freeze everything into tuples (immutability)
    frozen_levels: Tuple[Tuple[Config, ...], ...] = tuple(
        tuple(cfgs_n) for cfgs_n in levels_list
    )
    frozen_first_close: Tuple[Tuple[int, ...], ...] = tuple(
        tuple(fcs_n) for fcs_n in first_close_list
    )
    frozen_table_dn: Tuple[Tuple[Tuple[Config, ...], ...], ...] = tuple(
        tuple(
            tuple(cfgs_nd) for cfgs_nd in table_dn_n
        )
        for table_dn_n in table_dn_list
    )

    return NonCrossingPairingCache(
        N=N,
        max_d=max_d,
        levels=frozen_levels,
        first_close=frozen_first_close,
        table_dn=frozen_table_dn,
    )


################################################
# Public
################################################

def get_pairing_cache(N: int, max_d: Optional[int] = None) -> NonCrossingPairingCache:
    """
    Retrieve (or build) the non-crossing pairing cache for given N and max_d.

    - N >= 0 is the maximum n you will query.
    - If max_d is None, it defaults to N.

    Example
    -------
        cache = get_pairing_cache(6, max_d=6)
        cfgs = cache.table_dn[6][3]   # all configs on [1..6] valid for d=3
    """
    if max_d is None:
        max_d = N
    if N < 0:
        raise ValueError(f"N must be >= 0, got {N}")
    if max_d < 0:
        raise ValueError(f"max_d must be >= 0, got {max_d}")

    key = (N, max_d)
    if key not in _pairing_cache:
        _pairing_cache[key] = _build_cache(N, max_d)
    return _pairing_cache[key]


def saturated_pairings_for_d(n: int, d: int, max_d: Optional[int] = None) -> Tuple[Config, ...]:
    """
    Convenience function: return all saturated non-crossing matchings on [1..n]
    that satisfy the d-rule:

        "no pair (a,b) has a <= d and b <= d"

    with the convention that for n >= 1 and d > n, there is a single trivial
    configuration with all indices unmatched.

    This function builds (or reuses) a cache with N = n.

    Parameters
    ----------
    n : int
        Level n (interval [1..n]).

    d : int
        Restriction parameter d (0 <= d <= max_d).

    max_d : int or None
        Maximum d to build. If None, defaults to n.

    Returns
    -------
    Tuple[Config, ...]
        All configurations (pairs, unmatched) on [1..n] obeying the rule.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if max_d is None:
        max_d = n
    if d < 0 or d > max_d:
        raise ValueError(f"d must be in [0, max_d], got d={d}, max_d={max_d}")

    cache = get_pairing_cache(n, max_d=max_d)
    return cache.table_dn[n][d]


