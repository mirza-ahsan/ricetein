# Core cache quality check

The E1 tile cache and E1 whole-photo cache passed the declared finite/shape checks, exact repeated-description check, and comparison against saved descriptors. Both contain 3,355 inventory photos and eight versions (26,840 files each). Eligible supervised folds contain 3,334 photos; caching excluded photos is a per-photo operation and does not admit them to training. Source: `d44f65f`; exact fingerprints and counts are in `deadline_core_cache_checks.json`.

model_rater_c visually inspected the fixed, label-free E1 tile-neighbour gallery: six queries and four neighbours per query. This is a disclosed model inspection, not human annotation. The gallery contains background-only tiles and leaf/background boundaries. Neighbours generally share paper texture, background hue or diagonal leaf edges. No blank/corrupt rendering is apparent. The queries do not establish lesion-specific matching or disease semantics. Strong background matching is consistent with the shortcut diagnostics and must be reported as a limitation.

The gallery is retained locally at `cache/neighbour_galleries/e1_dinov3_vitb16.jpg`; photos are not committed. This check covers the two deadline core caches. Alternative caches, private feature publication/verification and full Stage 6 completion remain outstanding. It does not validate the later interpretability or Paddy experiments.
