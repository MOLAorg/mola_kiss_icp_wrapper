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

#include <mola_kernel/pretty_print_exception.h>
#include <mola_yaml/yaml_helpers.h>
#include <mrpt/3rdparty/tclap/CmdLine.h>
#include <mrpt/core/Clock.h>
#include <mrpt/core/exceptions.h>
#include <mrpt/io/lazy_load_path.h>
#include <mrpt/obs/CObservationPointCloud.h>
#include <mrpt/obs/CRawlog.h>
#include <mrpt/rtti/CObject.h>
#include <mrpt/system/COutputLogger.h>
#include <mrpt/system/datetime.h>
#include <mrpt/system/filesystem.h>
#include <mrpt/system/os.h>
#include <mrpt/system/progress.h>

#include <kiss_icp/pipeline/KissICP.hpp>

#if defined(HAVE_MOLA_INPUT_KITTI)
#include <mola_input_kitti_dataset/KittiOdometryDataset.h>
#endif

#include <csignal>  // sigaction
#include <cstdlib>
#include <iostream>
#include <string>

// Declare supported cli switches ===========
static TCLAP::CmdLine cmd("mola-lidar-odometry-cli-kiss");

static TCLAP::ValueArg<std::string> argYAML(
    "c", "config", "Input YAML config file (required) (*.yml)", false, "",
    "kiss-config.yml", cmd);

static TCLAP::ValueArg<std::string> arg_outPath(
    "", "output-tum-path",
    "Save the estimated path as a TXT file using the TUM file format (see evo "
    "docs)",
    false, "output-trajectory.txt", "output-trajectory.txt", cmd);

static TCLAP::ValueArg<int> arg_firstN(
    "", "only-first-n", "Run for the first N steps only (0=default, not used)",
    false, 0, "Number of dataset entries to run", cmd);

// Input dataset can come from one of these:
// --------------------------------------------
static TCLAP::ValueArg<std::string> argRawlog(
    "", "input-rawlog",
    "INPUT DATASET: rawlog. Input dataset in rawlog format (*.rawlog)", false,
    "dataset.rawlog", "dataset.rawlog", cmd);

#if defined(HAVE_MOLA_INPUT_KITTI)
static TCLAP::ValueArg<std::string> argKittiSeq(
    "", "input-kitti-seq",
    "INPUT DATASET: Use KITTI dataset sequence number 00|01|...", false, "00",
    "00", cmd);
static TCLAP::ValueArg<double> argKittiAngleDeg(
    "", "kitti-correction-angle-deg",
    "Correction vertical angle offset (see Deschaud,2018)", false, 0.205,
    "0.205 [degrees]", cmd);
#endif

class OfflineDatasetSource
{
   public:
    OfflineDatasetSource()          = default;
    virtual ~OfflineDatasetSource() = default;

    virtual size_t size() const = 0;

    virtual mrpt::obs::CObservation::Ptr getObservation(size_t index) const = 0;
};

class RawlogSource : public OfflineDatasetSource
{
   public:
    RawlogSource()          = default;
    virtual ~RawlogSource() = default;

    void init(const std::string& rawlogFile)
    {
        // Load dataset:
        std::cout << "Loading dataset: " << rawlogFile << std::endl;

        const auto imgsDir =
            mrpt::obs::CRawlog::detectImagesDirectory(rawlogFile);
        if (mrpt::system::directoryExists(imgsDir))
        {
            mrpt::io::setLazyLoadPathBase(imgsDir);
            std::cout << "Setting rawlog external directory to: " << imgsDir
                      << std::endl;
        }

        dataset_.loadFromRawLogFile(rawlogFile);
        std::cout << "Dataset loaded (" << dataset_.size() << " entries)."
                  << std::endl;
    }

    size_t size() const override { return dataset_.size(); }

    mrpt::obs::CObservation::Ptr getObservation(size_t index) const override
    {
        const mrpt::serialization::CSerializable::Ptr obj =
            dataset_.getAsGeneric(index);
        const auto obs =
            std::dynamic_pointer_cast<mrpt::obs::CObservation>(obj);
        return obs;
    }

   private:
    mrpt::obs::CRawlog dataset_;
};

#if defined(HAVE_MOLA_INPUT_KITTI)
class KittiSource : public OfflineDatasetSource
{
   public:
    KittiSource()          = default;
    virtual ~KittiSource() = default;

    void init(const std::string& kittiSeqNumber)
    {
        const auto kittiCfg =
            mola::Yaml::FromText(mola::parse_yaml(mrpt::format(
                R""""(
    params:
      base_dir: ${KITTI_BASE_DIR}
      sequence: '%s'
      time_warp_scale: 1.0
      publish_lidar: true
      publish_image_0: true
      publish_image_1: true
      publish_ground_truth: true
)"""",
                kittiSeqNumber.c_str())));

        kittiDataset_.initialize(kittiCfg);

        if (argKittiAngleDeg.isSet())
            kittiDataset_.VERTICAL_ANGLE_OFFSET =
                mrpt::DEG2RAD(argKittiAngleDeg.getValue());

        // Save GT, if available:
        if (arg_outPath.isSet() && kittiDataset_.hasGroundTruthTrajectory())
        {
            const auto& gtPath = kittiDataset_.getGroundTruthTrajectory();

            gtPath.saveToTextFile_TUM(
                mrpt::system::fileNameChangeExtension(
                    arg_outPath.getValue(), "") +
                std::string("_gt.txt"));
        }
    }

    size_t size() const override { return kittiDataset_.getTimestepCount(); }

    mrpt::obs::CObservation::Ptr getObservation(size_t index) const override
    {
        return kittiDataset_.getPointCloud(index);
    }

   private:
    mutable mola::KittiOdometryDataset kittiDataset_;
};

#endif

static int main_odometry()
{
    kiss_icp::pipeline::KISSConfig kissCfg;
    kissCfg.voxel_size = 1.0;

    kiss_icp::pipeline::KissICP kissIcp(kissCfg);

    // Select dataset input:
    std::shared_ptr<OfflineDatasetSource> dataset;

    if (argRawlog.isSet())
    {
        auto ds = std::make_shared<RawlogSource>();
        ds->init(argRawlog.getValue());

        dataset = ds;
    }
#if defined(HAVE_MOLA_INPUT_KITTI)
    else if (argKittiSeq.isSet())
    {
        auto ds = std::make_shared<KittiSource>();
        ds->init(argKittiSeq.getValue());
        dataset = ds;
    }
#endif
    else
    {
        THROW_EXCEPTION(
            "At least one of the dataset input CLI flags must be defined. "
            "Use --help.");
    }
    ASSERT_(dataset);

    const double tStart = mrpt::Clock::nowDouble();

    size_t nDatasetEntriesToRun = dataset->size();
    if (arg_firstN.isSet()) nDatasetEntriesToRun = arg_firstN.getValue();

    std::vector<mrpt::Clock::time_point> obsTimes;

    std::cout << "\n";  // Needed for the VT100 codes below.

    // Run:
    for (size_t i = 0; i < nDatasetEntriesToRun; i++)
    {
        const auto obs = dataset->getObservation(i);
        if (!obs) continue;

        // mrpt -> Eigen pointcloud
        auto obsPc =
            std::dynamic_pointer_cast<mrpt::obs::CObservationPointCloud>(obs);
        if (!obsPc) continue;
        ASSERT_(obsPc->pointcloud);

        std::vector<Eigen::Vector3d> inputPts;
        {
            const auto&  xs = obsPc->pointcloud->getPointsBufferRef_x();
            const auto&  ys = obsPc->pointcloud->getPointsBufferRef_y();
            const auto&  zs = obsPc->pointcloud->getPointsBufferRef_z();
            const size_t N  = xs.size();
            for (size_t j = 0; j < N; j++)
                inputPts.emplace_back(xs[j], ys[j], zs[j]);

            obsTimes.push_back(obsPc->timestamp);
        }

        kissIcp.RegisterFrame(inputPts);

        static int cnt = 0;
        if (cnt++ % 20 == 0)
        {
            cnt             = 0;
            const size_t N  = (dataset->size() - 1);
            const double pc = (1.0 * i) / N;

            const double tNow = mrpt::Clock::nowDouble();
            const double ETA  = pc > 0 ? (tNow - tStart) * (1.0 / pc - 1) : .0;

            std::cout << "\033[A\33[2KT\r"  // VT100 codes: up and clear line
                      << mrpt::system::progress(pc, 30)
                      << mrpt::format(
                             " %6zu/%6zu (%.02f%%) ETA=%s     \r", i, N,
                             100 * pc,
                             mrpt::system::formatTimeInterval(ETA).c_str());
            std::cout.flush();
        }
    }

    if (arg_outPath.isSet())
    {
        std::cout << "\nSaving estimated path in TUM format to: "
                  << arg_outPath.getValue() << std::endl;

        const auto                       path = kissIcp.poses();
        mrpt::poses::CPose3DInterpolator lastEstimatedTrajectory;
        for (size_t i = 0; i < path.size(); i++)
        {
            mrpt::poses::CPose3D pose =
                mrpt::poses::CPose3D::FromHomogeneousMatrix(path[i].matrix());
            lastEstimatedTrajectory.insert(obsTimes[i], pose);
        }

        lastEstimatedTrajectory.saveToTextFile_TUM(arg_outPath.getValue());
    }

    return 0;
}

int main(int argc, char** argv)
{
    try
    {
        // Parse arguments:
        if (!cmd.parse(argc, argv)) return 1;  // should exit.

        main_odometry();

        return 0;
    }
    catch (std::exception& e)
    {
        mola::pretty_print_exception(e, "Exit due to exception:");
        return 1;
    }
}