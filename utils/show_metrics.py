from evo.core import metrics
from evo.tools import log
log.configure_logging(verbose=True, debug=False, silent=False)
import numpy as np
from evo.tools import file_interface
from evo.core import sync
from evo.core import trajectory


def show_metrics(ref_file, est_file):
    traj_ref = file_interface.read_tum_trajectory_file(ref_file)
    traj_est = file_interface.read_tum_trajectory_file(est_file)
   
    max_diff = 0.01
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff)
    traj_est.align(traj_ref, correct_scale=True, correct_only_scale=False)
    data = (traj_ref, traj_est)

    #APE
    pose_relation = metrics.PoseRelation.translation_part
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    print("APE: ", round(ape_stat,4))
    print("--------------------------------------------------------")

    #RPE
    # normal mode
    deltas = range(100, 801, 100)
    delta_unit = metrics.Unit.meters
    seq_rpe = []
    all_pairs = True  # activate

    pose_relation = metrics.PoseRelation.translation_part
    for delta in deltas:
        try:
            rpe_metric = metrics.RPE(pose_relation, delta, delta_unit, 0.1, all_pairs)
            rpe_metric.process_data(data)
            rpe_stat = rpe_metric.get_statistic(metrics.StatisticsType.mean)
            seq_rpe.append(rpe_stat/delta)
        except:
            break
    rpe_trans = np.sum(seq_rpe)/len(seq_rpe)

    seq_rpe = []
    pose_relation = metrics.PoseRelation.rotation_angle_deg
    for delta in deltas:
        try:
            # all pairs mode
            rpe_metric = metrics.RPE(pose_relation, delta, delta_unit, 0.1, all_pairs)
            rpe_metric.process_data(data)
            rpe_stat = rpe_metric.get_statistic(metrics.StatisticsType.mean)
            seq_rpe.append(rpe_stat)#/delta)
        except:
            break

    rpe_rot = np.sum(seq_rpe)/len(seq_rpe)
    print("--------------------------------------------------------")
    print("RPE: trans = ", round(rpe_trans*100,4),"% rot = ", round(rpe_rot,4), "deg/m")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_file", help="Ground truth file (it needs to be a in TUM format)")
    parser.add_argument("est_file", help="Odometry estimation (it needs to be a in TUM format)")
    args = parser.parse_args()
    show_metrics(args.ref_file, args.est_file)
