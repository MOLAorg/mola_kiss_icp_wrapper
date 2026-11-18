# mola_kiss_icp_wrapper
Wrapper to use Kiss ICP on MOLA dataset sources via CLI

KISS-ICP's C++ core is built from the submodule `submodules/kiss-icp`, branch
`v1.3.0-mola` of `MOLAorg/kiss-icp-optional-tbb` (upstream KISS-ICP v1.3.0).

## Usage

    mola-lidar-odometry-cli-kiss --input-rawlog dataset.rawlog \
      --lidar-sensor-label lidar1 --min-range 0.5 --max-range 60 \
      --output-tum-path trajectory.tum

Other inputs: `--input-rosbag2`, `--input-rosbag1`, `--input-kitti-seq`,
`--input-kitti360-seq`, `--input-mulran-seq`, `--input-paris-luco`.
See `--help`.

## KISS-ICP parameters

| CLI option                | Default         | KISS-ICP parameter                     |
|---------------------------|-----------------|----------------------------------------|
| `--min-range`             | 0               | `data.min_range`                       |
| `--max-range`             | 100             | `data.max_range`                       |
| `--no-deskew`             | (de-skew on)    | `data.deskew`                          |
| `--voxel-size`            | max-range / 100 | `mapping.voxel_size`                   |
| `--max-points-per-voxel`  | 20              | `mapping.max_points_per_voxel`         |
| `--initial-threshold`     | 2.0             | `adaptive_threshold.initial_threshold` |
| `--min-motion-th`         | 0.1             | `adaptive_threshold.min_motion_th`     |
| `--max-num-iterations`    | 500             | `registration.max_num_iterations`      |
| `--convergence-criterion` | 0.0001          | `registration.convergence_criterion`   |
| `--num-threads`           | 0 (all cores)   | `registration.max_num_threads`         |

`--num-threads 0` uses every core the process may run on (its CPU affinity
mask). Use `--num-threads 1`, or pin the process to one core, for repeatable
results.

De-skew uses the per-point time field of the scans, if present. A de-skewed
scan is expressed at the time of its latest point, so each pose of the output
trajectory is stamped at that time. Without de-skew, it is stamped at the
observation timestamp.

## Changes from the previous KISS-ICP version (0.3)

No CLI option was renamed. Changes:

- `--min-range` now defaults to 0, as upstream (it was 5).
- New options: `--max-num-iterations`, `--convergence-criterion`,
  `--num-threads`. Their defaults are the values that were hard-coded before,
  and TBB's default number of threads.
- Output stamps of de-skewed scans: the time of the latest point of the scan.
  Before, they were the observation timestamp, while the old de-skew expressed
  the scan at its mid time.
- Upstream algorithm changes: the robust kernel scale is the adaptive
  threshold (it was a third of it); the initial threshold is kept as one
  sample of the adaptive threshold instead of being used until the sensor
  moves; de-skew uses the last relative motion and the scan end as reference;
  the voxel map keeps a minimum distance between the points of a voxel; voxel
  indices are rounded down (they were truncated toward zero).
