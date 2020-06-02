#
# MIT License
#
# Copyright (c) 2020 Keisuke Sehara
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""a thin wrapper for the DLC pose-estimation routine."""

from collections import namedtuple as _namedtuple
from pathlib import Path as _Path

import numpy as _np
import tensorflow as _tf

from deeplabcut.utils import auxiliaryfunctions as _aux
from deeplabcut.pose_estimation_tensorflow.nnet import predict as _predict

vers = (_tf.__version__).split('.')
if int(vers[0])==1 and int(vers[1])>12:
    TF = _tf.compat.v1
else:
    TF = _tf

class TFSession(_namedtuple('_TFSession', ('config', 'session', 'input', 'output'))):
    @classmethod
    def from_config(cls, cfg, gputouse=None, shuffle=1, trainIndex=0):
        return init_session(cfg, gputouse=gputouse, shuffle=shuffle, trainIndex=trainIndex)

    def get_pose(self, image, outall=False):
        '''returns pose=(part1x, part1y, part1prob, part2x, part2y, ...)'''
        image = _np.expand_dims(image, axis=0).astype(float)
        cfg, sess, inputs, outputs = self
        outputs_np = sess.run(outputs, feed_dict={inputs: image})
        scmap, locref = _predict.extract_cnn_output(outputs_np, cfg)
        pose = _predict.argmax_pose_predict(scmap, locref, cfg.stride)
        if outall:
            return scmap, locref, pose
        else:
            return pose

def _get_pose_config(cfg, modelfolder, shuffle=1, trainIndex=0):
    from deeplabcut.pose_estimation_tensorflow import config as _config
    projpath      = _Path(cfg["project_path"])
    pose_file     = modelfolder / 'test' / 'pose_cfg.yaml'
    try:
        return _config.load_config(str(pose_file))
    except FileNotFoundError:
        raise FileNotFoundError(f"'pose_cfg.yaml' for shuffle {shuffle} trainFraction {cfg['TrainingFraction'][trainIndex]}.")

def _get_snapshot(cfg, modelfolder, shuffle=1):
    # Check which snapshots are available and sort them by # iterations
    traindir = modelfolder / 'train'
    def __iteration(snapshot):
        return int(snapshot.stem.split('-')[1])
    available_snapshots = sorted([snapshot for snapshot in traindir.glob('*.index')], key=__iteration)
    if len(available_snapshots) == 0:
      raise ValueError(f"Snapshots not found! It seems the dataset for shuffle {shuffle} has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle {shuffle}.")

    if cfg['snapshotindex'] == 'all':
        print("Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!")
        snapshotindex = -1
    else:
        snapshotindex=cfg['snapshotindex']
    if abs(snapshotindex) >= len(available_snapshots):
        raise IndexError(f"invalid index {snapshotindex} for {len(available_snapshots)} snapshots")

    snapshot = available_snapshots[snapshotindex].with_suffix('')
    print("Using %s" % snapshot, "for model", modelfolder)
    return snapshot, __iteration(snapshot)

def init_session(cfg, gputouse=None, shuffle=1, trainIndex=0):
    if isinstance(cfg, (str, _Path)):
        cfg = _aux.read_config(str(cfg))
    TF.reset_default_graph()

    projpath      = cfg['project_path']
    trainFraction = cfg['TrainingFraction'][trainIndex]
    modelfolder   = projpath / _aux.GetModelFolder(trainFraction,shuffle,cfg)
    dlc_cfg       = _get_pose_config(cfg, modelfolder, shuffle=shuffle, trainIndex=trainIndex)
    snapshot, iteration = _get_snapshot(cfg, modelfolder, shuffle=shuffle)

    dlc_cfg['init_weights'] = str(snapshot)
    #update batchsize (based on parameters in config.yaml)
    dlc_cfg['batch_size'] = cfg['batch_size']
    # update number of outputs
    dlc_cfg['num_outputs'] = cfg.get('num_outputs', 1)
    print('num_outputs = ', dlc_cfg['num_outputs'])
    DLCscorer = _aux.GetScorerName(cfg,shuffle,trainFraction,trainingsiterations=iteration)

    return TFSession(dlc_cfg, *(_predict.setup_pose_prediction(dlc_cfg)))
