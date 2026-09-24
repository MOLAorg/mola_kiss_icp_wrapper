/* -------------------------------------------------------------------------
 *   A Modular Optimization framework for Localization and mApping  (MOLA)
 *
 * Copyright (C) 2018-2023 Jose Luis Blanco, University of Almeria
 * Licensed under the GNU GPL v3 for non-commercial applications.
 *
 * This file is part of MOLA.
 * MOLA is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * MOLA is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * MOLA. If not, see <https://www.gnu.org/licenses/>.
 * ------------------------------------------------------------------------- */

/**
 * @file   mola-lidar-odometry-cli-kiss.cpp
 * @brief  main() for the cli app wrapping kiss-icp for MOLA data inputs.
 * @author Jose Luis Blanco Claraco
 * @date   Sep 22, 2023
 */

#include <mola_kernel/interfaces/OfflineDatasetSource.h>
#include <mola_kernel/pretty_print_exception.h>
#include <mola_yaml/yaml_helpers.h>
#include <mp2p_icp_filters/Generator.h>
#include <mrpt/core/Clock.h>
#include <mrpt/core/exceptions.h>
#include <mrpt/io/lazy_load_path.h>
#include <mrpt/maps/CSimplePointsMap.h>
#include <mrpt/obs/CObservation2DRangeScan.h>
#include <mrpt/obs/CObservation3DRangeScan.h>
#include <mrpt/obs/CObservationPointCloud.h>
#include <mrpt/obs/CObservationRotatingScan.h>
#include <mrpt/obs/CObservationVelodyneScan.h>
#include <mrpt/obs/CRawlog.h>
#include <mrpt/poses/CPose3DInterpolator.h>
#include <mrpt/rtti/CObject.h>
#include <mrpt/system/COutputLogger.h>
#include <mrpt/system/datetime.h>
#include <mrpt/system/filesystem.h>
#include <mrpt/system/os.h>
#include <mrpt/system/progress.h>
#include <mrpt/system/string_utils.h>
#include <mrpt/version.h>

#include <CLI/CLI.hpp>
#include <kiss_icp/pipeline/KissICP.hpp>

#if defined(HAVE_MOLA_INPUT_KITTI)
#include <mola_input_kitti_dataset/KittiOdometryDataset.h>
#endif

#if defined(HAVE_MOLA_INPUT_KITTI360)
#include <mola_input_kitti360_dataset/Kitti360Dataset.h>
#endif

#if defined(HAVE_MOLA_INPUT_MULRAN)
#include <mola_input_mulran_dataset/MulranDataset.h>
#endif

#if defined(HAVE_MOLA_INPUT_RAWLOG)
#include <mola_input_rawlog/RawlogDataset.h>
#endif

#if defined(HAVE_MOLA_INPUT_ROSBAG2)
#include <mola_input_rosbag2/Rosbag2Dataset.h>
#endif

#if defined(HAVE_MOLA_INPUT_ROSBAG1)
#include <mola_input_rosbag1/Rosbag1Dataset.h>
#endif

#if defined(HAVE_MOLA_INPUT_PARIS_LUCO)
#include <mola_input_paris_luco_dataset/ParisLucoDataset.h>
#endif

#include <cstdlib>
#include <iostream>
#include <string>

// Declare supported cli switches ===========
static CLI::App cmd{"mola-lidar-odometry-cli-kiss"};

static double argMinRange{5.0};
static bool   argMinRange_set{false};

static double argMaxRange{100.0};
static bool   argMaxRange_set{false};

static std::string arg_outPath{"output-trajectory.txt"};
static bool        arg_outPath_set{false};

static bool argNoDeskew{false};

static int  arg_firstN{0};
static bool arg_firstN_set{false};

// Input dataset can come from one of these:
// --------------------------------------------
#if defined(HAVE_MOLA_INPUT_RAWLOG)
static std::string argRawlog{"dataset.rawlog"};
static bool        argRawlog_set{false};
#endif

#if defined(HAVE_MOLA_INPUT_ROSBAG2)
static std::string argRosbag2{"dataset.mcap"};
static bool        argRosbag2_set{false};
#endif

#if defined(HAVE_MOLA_INPUT_ROSBAG1)
static std::string argRosbag1{"dataset.bag"};
static bool        argRosbag1_set{false};
#endif

static std::string arg_lidarLabel;
static bool        arg_lidarLabel_set{false};

#if defined(HAVE_MOLA_INPUT_KITTI)
static std::string argKittiSeq{"00"};
static bool        argKittiSeq_set{false};
static double      argKittiAngleDeg{0.205};
static bool        argKittiAngleDeg_set{false};
#endif

#if defined(HAVE_MOLA_INPUT_KITTI360)
static std::string argKitti360Seq{"00"};
static bool        argKitti360Seq_set{false};
#endif

#if defined(HAVE_MOLA_INPUT_MULRAN)
static std::string argMulranSeq{"KAIST01"};
static bool        argMulranSeq_set{false};
#endif

#if defined(HAVE_MOLA_INPUT_PARIS_LUCO)
static bool argParisLucoSeq{false};
#endif

#if defined(HAVE_MOLA_INPUT_RAWLOG)
std::shared_ptr<mola::OfflineDatasetSource> dataset_from_rawlog(
    const std::string& rawlogFile)
{
    auto o = std::make_shared<mola::RawlogDataset>();

    const auto cfg = mola::Yaml::FromText(mola::parse_yaml(mrpt::format(
        R""""(
    params:
      rawlog_filename: '%s'
      read_all_first: true
)"""",
        rawlogFile.c_str())));

    o->initialize(cfg);

    return o;
}
#endif

#if defined(HAVE_MOLA_INPUT_KITTI)
std::shared_ptr<mola::OfflineDatasetSource> dataset_from_kitti(
    const std::string& kittiSeqNumber)
{
    auto o = std::make_shared<mola::KittiOdometryDataset>();

    const auto cfg = mola::Yaml::FromText(mola::parse_yaml(mrpt::format(
        R""""(
    params:
      base_dir: ${KITTI_BASE_DIR}
      sequence: '%s'
      time_warp_scale: 1.0
      clouds_as_organized_points: false
      publish_lidar: true
      publish_image_0: false
      publish_image_1: false
      publish_ground_truth: true
)"""",
        kittiSeqNumber.c_str())));

    o->initialize(cfg);

    if (argKittiAngleDeg_set)
        o->VERTICAL_ANGLE_OFFSET = mrpt::DEG2RAD(argKittiAngleDeg);

    // Save GT, if available:
    if (arg_outPath_set && o->hasGroundTruthTrajectory())
    {
        const auto& gtPath = o->getGroundTruthTrajectory();

        gtPath.saveToTextFile_TUM(
            mrpt::system::fileNameChangeExtension(arg_outPath, "") +
            std::string("_gt.txt"));
    }

    return o;
}
#endif

#if defined(HAVE_MOLA_INPUT_KITTI360)
std::shared_ptr<mola::OfflineDatasetSource> dataset_from_kitti360(
    const std::string& kittiSeqNumber)
{
    auto o = std::make_shared<mola::Kitti360Dataset>();

    const auto cfg = mola::Yaml::FromText(mola::parse_yaml(mrpt::format(
        R""""(
    params:
      base_dir: ${KITTI360_DATASET}
      sequence: '%s'
      time_warp_scale: 1.0
      clouds_as_organized_points: false
      publish_lidar: true
      publish_image_0: false
      publish_image_1: false
      publish_ground_truth: true
)"""",
        kittiSeqNumber.c_str())));

    o->initialize(cfg);

    // Save GT, if available:
    if (arg_outPath_set && o->hasGroundTruthTrajectory())
    {
        const auto& gtPath = o->getGroundTruthTrajectory();

        gtPath.saveToTextFile_TUM(
            mrpt::system::fileNameChangeExtension(arg_outPath, "") +
            std::string("_gt.txt"));
    }

    return o;
}
#endif

#if defined(HAVE_MOLA_INPUT_ROSBAG2)
std::shared_ptr<mola::OfflineDatasetSource> dataset_from_rosbag2(
    const std::string& rosbag2file)
{
    ASSERTMSG_(
        arg_lidarLabel_set,
        "Using a rosbag2 as input requires telling what is the lidar topic "
        "with --lidar-sensor-label <TOPIC_NAME>");

    auto o = std::make_shared<mola::Rosbag2Dataset>();

    // A comma-separated value becomes a YAML sequence, so a recording split
    // across several bag directories (e.g. Oxford Spires keble-college-04,
    // two halves of one continuous recording) is replayed as the single
    // sequence it is. Rosbag2Dataset accepts a scalar or a sequence. Same
    // handling, and the same spelling of the input, as the sibling
    // mola-lidar-odometry-cli, so one caller can drive either binary.
    std::string bagsYaml;
    {
        std::vector<std::string> parts;
        mrpt::system::tokenize(rosbag2file, ",", parts);
        ASSERT_(!parts.empty());
        if (parts.size() == 1)
        {
            bagsYaml = "'" + mrpt::system::trim(parts[0]) + "'";
        }
        else
        {
            for (const auto& bp : parts)
                bagsYaml += "\n        - '" + mrpt::system::trim(bp) + "'";
        }
    }

    const auto cfg = mola::Yaml::FromText(mola::parse_yaml(mrpt::format(
        R""""(
    params:
      rosbag_filename: %s
      base_link_frame_id: "${MOLA_TF_BASE_LINK|base_footprint}"
      sensors:
        - topic: '%s'
          type: CObservationPointCloud
          # Same env var names as the sibling MOLA odometry CLIs, so one
          # override snippet serves every method. Bags without /tf need this.
          fixed_sensor_pose: "${LIDAR_POSE_X|0} ${LIDAR_POSE_Y|0} ${LIDAR_POSE_Z|0} ${LIDAR_POSE_YAW|0} ${LIDAR_POSE_PITCH|0} ${LIDAR_POSE_ROLL|0}"
          # Defaults to true, which is this CLI's long-standing behavior here.
          use_fixed_sensor_pose: ${MOLA_USE_FIXED_LIDAR_POSE|true}
)"""",
        bagsYaml.c_str(), arg_lidarLabel.c_str())));

    o->initialize(cfg);

    return o;
}
#endif

#if defined(HAVE_MOLA_INPUT_ROSBAG1)
std::shared_ptr<mola::OfflineDatasetSource> dataset_from_rosbag1(
    const std::string& rosbag1file)
{
    ASSERTMSG_(
        arg_lidarLabel_set,
        "Using a rosbag1 as input requires telling what is the lidar topic "
        "with --lidar-sensor-label <TOPIC_NAME>");

    auto o = std::make_shared<mola::Rosbag1Dataset>();

    // A comma-separated value becomes a YAML sequence, so a recording split
    // across several bag files is replayed as the single sequence it is.
    // Rosbag1Dataset accepts either a scalar or a sequence. This matters more
    // here than for rosbag2: the ROS 1-era datasets in this corpus are split
    // into fixed-size chunks, up to sixteen of them for one sequence.
    std::string bagsYaml;
    {
        std::vector<std::string> parts;
        mrpt::system::tokenize(rosbag1file, ",", parts);
        ASSERT_(!parts.empty());
        if (parts.size() == 1)
        {
            bagsYaml = "'" + mrpt::system::trim(parts[0]) + "'";
        }
        else
        {
            for (const auto& p : parts)
                bagsYaml += "\n        - '" + mrpt::system::trim(p) + "'";
        }
    }

    // Only a lidar entry: KISS-ICP is pure LiDAR odometry, with no IMU input
    // at all, so an imu sensor entry here would be read from the bag and
    // then dropped. Same env var names as dataset_from_rosbag2() above, so
    // one override snippet -- and the same dataset profiles -- serve both.
    const auto cfg = mola::Yaml::FromText(mola::parse_yaml(mrpt::format(
        R""""(
    params:
      rosbag_filename: %s
      base_link_frame_id: "${MOLA_TF_BASE_LINK|base_footprint}"
      sensors:
        - topic: '%s'
          type: CObservationPointCloud
          fixed_sensor_pose: "${LIDAR_POSE_X|0} ${LIDAR_POSE_Y|0} ${LIDAR_POSE_Z|0} ${LIDAR_POSE_YAW|0} ${LIDAR_POSE_PITCH|0} ${LIDAR_POSE_ROLL|0}"
          # Defaults to true, matching dataset_from_rosbag2() above rather
          # than mola-lidar-odometry-cli's false: the two CLIs have always
          # disagreed on this default, and the dataset profiles set it
          # explicitly, which is what keeps every method on one rig.
          use_fixed_sensor_pose: ${MOLA_USE_FIXED_LIDAR_POSE|true}
)"""",
        bagsYaml.c_str(), arg_lidarLabel.c_str())));

    o->initialize(cfg);

    return o;
}
#endif

#if defined(HAVE_MOLA_INPUT_MULRAN)
std::shared_ptr<mola::OfflineDatasetSource> dataset_from_mulran(
    const std::string& mulranSequence)
{
    auto o = std::make_shared<mola::MulranDataset>();

    const auto cfg = mola::Yaml::FromText(mola::parse_yaml(mrpt::format(
        R""""(
    params:
      base_dir: ${MULRAN_BASE_DIR}
      sequence: '%s'
      time_warp_scale: 1.0
      publish_lidar: true
      publish_ground_truth: true
)"""",
        mulranSequence.c_str())));

    o->initialize(cfg);

    return o;
}
#endif

#if defined(HAVE_MOLA_INPUT_PARIS_LUCO)
std::shared_ptr<mola::OfflineDatasetSource> dataset_from_paris_luco()
{
    auto o = std::make_shared<mola::ParisLucoDataset>();

    const auto cfg = mola::Yaml::FromText(mola::parse_yaml(
        R""""(
    params:
      base_dir: ${PARIS_LUCO_BASE_DIR}
      sequence: '00'  # There is only one sequence in this dataset
      time_warp_scale: 1.0
)""""));

    o->initialize(cfg);

    return o;
}
#endif

static int main_odometry()
{
    kiss_icp::pipeline::KISSConfig kissCfg;
    kissCfg.voxel_size = 1.0;
    kissCfg.deskew     = true;

    if (argMinRange_set) kissCfg.min_range = argMinRange;
    if (argMaxRange_set) kissCfg.max_range = argMaxRange;
    if (argNoDeskew) kissCfg.deskew = false;

    kiss_icp::pipeline::KissICP kissIcp(kissCfg);

    // Select dataset input:
    std::shared_ptr<mola::OfflineDatasetSource> dataset;

#if defined(HAVE_MOLA_INPUT_RAWLOG)
    if (argRawlog_set) { dataset = dataset_from_rawlog(argRawlog); }
    else
#endif
#if defined(HAVE_MOLA_INPUT_KITTI)
        if (argKittiSeq_set)
    {
        dataset = dataset_from_kitti(argKittiSeq);
    }
    else
#endif
#if defined(HAVE_MOLA_INPUT_KITTI360)
        if (argKitti360Seq_set)
    {
        dataset = dataset_from_kitti360(argKitti360Seq);
    }
    else
#endif
#if defined(HAVE_MOLA_INPUT_MULRAN)
        if (argMulranSeq_set)
    {
        dataset = dataset_from_mulran(argMulranSeq);
    }
    else
#endif
#if defined(HAVE_MOLA_INPUT_ROSBAG2)
        if (argRosbag2_set)
    {
        dataset = dataset_from_rosbag2(argRosbag2);
    }
    else
#endif
#if defined(HAVE_MOLA_INPUT_ROSBAG1)
        if (argRosbag1_set)
    {
        dataset = dataset_from_rosbag1(argRosbag1);
    }
    else
#endif
#if defined(HAVE_MOLA_INPUT_PARIS_LUCO)
        if (argParisLucoSeq)
    {
        dataset = dataset_from_paris_luco();
    }
    else
#endif
    {
        THROW_EXCEPTION(
            "At least one of the dataset input CLI flags must be defined. "
            "Use --help.");
    }
    ASSERT_(dataset);

    const double tStart = mrpt::Clock::nowDouble();

    size_t nDatasetEntriesToRun = dataset->datasetSize();
    if (arg_firstN_set) nDatasetEntriesToRun = arg_firstN;

    mp2p_icp_filters::Generator g;
    g.initialize({});

    std::vector<mrpt::Clock::time_point> obsTimes;

    std::cout << "\n";  // Needed for the VT100 codes below.

    // Run:
    for (size_t i = 0; i < nDatasetEntriesToRun; i++)
    {
        // Get observations from the dataset:
        using namespace mrpt::obs;

        const auto sf = dataset->datasetGetObservations(i);
        ASSERT_(sf);

        CObservation::Ptr obs;
        obs = sf->getObservationByClass<CObservationRotatingScan>();
        if (!obs) obs = sf->getObservationByClass<CObservationPointCloud>();
        if (!obs) obs = sf->getObservationByClass<CObservation3DRangeScan>();
        if (!obs) obs = sf->getObservationByClass<CObservation2DRangeScan>();
        if (!obs) obs = sf->getObservationByClass<CObservationVelodyneScan>();

        if (!obs) continue;

        if (!arg_lidarLabel.empty() && obs->sensorLabel != arg_lidarLabel)
            continue;

        // mrpt -> Eigen pointcloud
        std::vector<Eigen::Vector3d> inputPts;
        std::vector<double>          inputPtTimestamps;

        auto lmbPcToPoints = [&](const mrpt::maps::CPointsMap& pc)
        {
            const auto& xs = pc.getPointsBufferRef_x();
            const auto& ys = pc.getPointsBufferRef_y();
            const auto& zs = pc.getPointsBufferRef_z();

#if MRPT_VERSION >= 0x020f00  // 2.15.0
            auto* Ts = pc.getPointsBufferRef_float_field(
                mrpt::maps::CPointsMap::POINT_FIELD_TIMESTAMP);
#else
            const auto* Ts = pc.getPointsBufferRef_timestamp();  // optional
#endif
            const size_t N = xs.size();

            for (size_t j = 0; j < N; j++)
            {
                inputPts.emplace_back(xs[j], ys[j], zs[j]);
            }

            if (Ts && !Ts->empty())
            {
                ASSERT_(Ts->size() == N);

                // KISS ICP assumes times in the range [0,1]:

                const float t0 = *std::min_element(Ts->cbegin(), Ts->cend());
                const float t1 = *std::max_element(Ts->cbegin(), Ts->cend());

                // A constant time field (e.g. instantaneous, simulated
                // scans) carries no timing: handle it as a missing one.
                if (t1 > t0)
                {
                    const float k = 1.0f / (t1 - t0);

                    for (size_t j = 0; j < N; j++)
                        inputPtTimestamps.emplace_back(((*Ts)[j] - t0) * k);
                }
            }
        };

        // Load lazy-load obs:
        obs->load();

        // generic conversion to point cloud:
        {
            mp2p_icp::metric_map_t mm;
            g.process(*obs, mm);
            const auto rawLayer = mm.point_layer("raw");
            ASSERT_(rawLayer);
            lmbPcToPoints(*rawLayer);
        }

        if (inputPts.empty()) continue;

        // Pushed here, next to the RegisterFrame() that produces the matching
        // pose, and NOT while converting the cloud: a scan that converts to
        // zero points is skipped by the `continue` above without adding a
        // pose, so recording its timestamp earlier left obsTimes one entry
        // ahead of kissIcp.poses() and shifted the timestamp of every later
        // pose by one scan -- silently, since the trajectory still looked
        // well-formed. The two containers now grow together by construction,
        // which the assertion after the loop re-checks.
        obsTimes.push_back(obs->timestamp);

        if (inputPtTimestamps.empty())
            kissIcp.RegisterFrame(inputPts);
        else
            kissIcp.RegisterFrame(inputPts, inputPtTimestamps);

        static int cnt = 0;
        if (cnt++ % 20 == 0)
        {
            cnt             = 0;
            const size_t N  = (dataset->datasetSize() - 1);
            const double pc = (1.0 * i) / N;

            const double tNow = mrpt::Clock::nowDouble();
            const double ETA  = pc > 0 ? (tNow - tStart) * (1.0 / pc - 1) : .0;
            const double totalTime = ETA + (tNow - tStart);

            std::cout
                << "\033[A\33[2KT\r"  // VT100 codes: up and clear line
                << mrpt::system::progress(pc, 30)
                << mrpt::format(
                       " %6zu/%6zu (%.02f%%) ETA=%s / T=%s\n", i, N, 100 * pc,
                       mrpt::system::formatTimeInterval(ETA).c_str(),
                       mrpt::system::formatTimeInterval(totalTime).c_str());

            std::cout.flush();
        }
    }

    if (arg_outPath_set)
    {
        std::cout << "\nSaving estimated path in TUM format to: " << arg_outPath
                  << std::endl;

        const auto                       path = kissIcp.poses();
        mrpt::poses::CPose3DInterpolator lastEstimatedTrajectory;

        // One timestamp per registered frame, one pose per registered frame.
        // Checked rather than assumed: indexing obsTimes by pose number is
        // only correct while that holds, and reading past its end would be
        // undefined behavior rather than a visible failure.
        ASSERT_EQUAL_(path.size(), obsTimes.size());
        for (size_t i = 0; i < path.size(); i++)
        {
            mrpt::poses::CPose3D pose =
                mrpt::poses::CPose3D::FromHomogeneousMatrix(path[i].matrix());
            lastEstimatedTrajectory.insert(obsTimes[i], pose);
        }

        lastEstimatedTrajectory.saveToTextFile_TUM(arg_outPath);
    }

    return 0;
}

int main(int argc, char** argv)
{
    try
    {
        auto* optMinRange =
            cmd.add_option("--min-range", argMinRange, "min-range parameter");
        auto* optMaxRange =
            cmd.add_option("--max-range", argMaxRange, "max-range parameter");
        auto* optOutPath = cmd.add_option(
            "--output-tum-path", arg_outPath,
            "Save the estimated path as a TXT file using the TUM file format "
            "(see evo "
            "docs)");
        cmd.add_flag("--no-deskew", argNoDeskew, "Skip scan de-skew");
        auto* optFirstN = cmd.add_option(
            "--only-first-n", arg_firstN,
            "Run for the first N steps only (0=default, not used)");

#if defined(HAVE_MOLA_INPUT_RAWLOG)
        auto* optRawlog = cmd.add_option(
            "--input-rawlog", argRawlog,
            "INPUT DATASET: rawlog. Input dataset in rawlog format (*.rawlog)");
#endif

#if defined(HAVE_MOLA_INPUT_ROSBAG2)
        auto* optRosbag2 = cmd.add_option(
            "--input-rosbag2", argRosbag2,
            "INPUT DATASET: rosbag2. Input dataset in rosbag2 format (*.mcap)");
#endif

#if defined(HAVE_MOLA_INPUT_ROSBAG1)
        auto* optRosbag1 = cmd.add_option(
            "--input-rosbag1", argRosbag1,
            "INPUT DATASET: rosbag1. Input dataset in ROS 1 bag format "
            "(*.bag). "
            "Accepts several comma-separated files, replayed as one "
            "chronological "
            "sequence");
#endif

        auto* optLidarLabel = cmd.add_option(
            "--lidar-sensor-label", arg_lidarLabel,
            "If provided, this supersedes the values in the "
            "'lidar_sensor_labels' "
            "entry of the odometry pipeline, defining the sensorLabel/topic "
            "name to "
            "read LIDAR data from. It can be a regular expression "
            "(std::regex)");

#if defined(HAVE_MOLA_INPUT_KITTI)
        auto* optKittiSeq = cmd.add_option(
            "--input-kitti-seq", argKittiSeq,
            "INPUT DATASET: Use KITTI dataset sequence number 00|01|...");
        auto* optKittiAngleDeg = cmd.add_option(
            "--kitti-correction-angle-deg", argKittiAngleDeg,
            "Correction vertical angle offset (see Deschaud,2018)");
#endif

#if defined(HAVE_MOLA_INPUT_KITTI360)
        auto* optKitti360Seq = cmd.add_option(
            "--input-kitti360-seq", argKitti360Seq,
            "INPUT DATASET: Use KITTI360 dataset sequence number 00|01|...");
#endif

#if defined(HAVE_MOLA_INPUT_MULRAN)
        auto* optMulranSeq = cmd.add_option(
            "--input-mulran-seq", argMulranSeq,
            "INPUT DATASET: Use Mulran dataset sequence KAIST01|KAIST01|...");
#endif

#if defined(HAVE_MOLA_INPUT_PARIS_LUCO)
        cmd.add_flag(
            "--input-paris-luco", argParisLucoSeq,
            "INPUT DATASET: Use Paris Luco dataset (unique sequence=00)");
#endif

        CLI11_PARSE(cmd, argc, argv);

        argMinRange_set = (optMinRange->count() > 0);
        argMaxRange_set = (optMaxRange->count() > 0);
        arg_outPath_set = (optOutPath->count() > 0);
        arg_firstN_set  = (optFirstN->count() > 0);
#if defined(HAVE_MOLA_INPUT_RAWLOG)
        argRawlog_set = (optRawlog->count() > 0);
#endif
#if defined(HAVE_MOLA_INPUT_ROSBAG2)
        argRosbag2_set = (optRosbag2->count() > 0);
#endif
#if defined(HAVE_MOLA_INPUT_ROSBAG1)
        argRosbag1_set = (optRosbag1->count() > 0);
#endif
        arg_lidarLabel_set = (optLidarLabel->count() > 0);
#if defined(HAVE_MOLA_INPUT_KITTI)
        argKittiSeq_set      = (optKittiSeq->count() > 0);
        argKittiAngleDeg_set = (optKittiAngleDeg->count() > 0);
#endif
#if defined(HAVE_MOLA_INPUT_KITTI360)
        argKitti360Seq_set = (optKitti360Seq->count() > 0);
#endif
#if defined(HAVE_MOLA_INPUT_MULRAN)
        argMulranSeq_set = (optMulranSeq->count() > 0);
#endif

        main_odometry();

        return 0;
    }
    catch (std::exception& e)
    {
        mola::pretty_print_exception(e, "Exit due to exception:");
        return 1;
    }
}
