#
# MIT License
#
# Copyright (c) 2020-2021 Keisuke Sehara
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
import warnings as _warnings

import numpy as _np

SHOW_WARNINGS = False

if SHOW_WARNINGS == False:
    _warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")
    _warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")
import tensorflow as _tf
from deeplabcut.utils import auxiliaryfunctions as _aux
from deeplabcut.pose_estimation_tensorflow.nnet.net_factory import pose_net as _pose_net
from deeplabcut.pose_estimation_tensorflow import config as _config

from . import load_config as _load_config

vers = (_tf.__version__).split('.')
if int(vers[0])==1 and int(vers[1])>12:
    TF = _tf.compat.v1
else:
    TF = _tf

class TFSession(_namedtuple('_TFSession',
                                ('config',
                                 'session',
                                 'input',
                                 'output',
                                 'locate_on_gpu'))):
    @classmethod
    def from_config(cls, cfg, gputouse=None, shuffle=1, trainIndex=0, locate_on_gpu=False):
        return init_session(cfg,
                            gputouse=gputouse,
                            shuffle=shuffle,
                            trainIndex=trainIndex,
                            locate_on_gpu=locate_on_gpu)

    def open_writer(self, path, sep=","):
        return _writer.CSVWriter(path, tfsession=self, sep=sep)

class DirectTFSession(TFSession):
    def get_pose(self, image):
        '''returns pose=(part1x, part1y, part1prob, part2x, part2y, ...)'''
        image = _np.expand_dims(image, axis=0).astype(float)
        cfg, sess, inputs, outputs, locate_on_gpu = self
        outputs_np = sess.run(outputs, feed_dict={inputs: image})
        if locate_on_gpu == True:
            return outputs_np[0]
        else:
            scmap, locref = _dlc_extract_cnn_output(outputs_np, cfg)
            pose = _dlc_argmax_pose_predict(scmap, locref, cfg["stride"])
            #if outall:
            #    return scmap, locref, pose
            #else:
            return pose

class BatchTFSession(TFSession):
    def get_pose(self, images):
        '''returns pose=(part1x, part1y, part1prob, part2x, part2y, ...)'''
        if not isinstance(images, _np.ndarray):
            images = _np.stack(images, axis=0)
        images = images.astype(float)
        cfg, sess, inputs, outputs, locate_on_gpu = self
        outputs_np = sess.run(outputs, feed_dict={inputs: images})
        if locate_on_gpu == True:
            return outputs_np[0].reshape((cfg.batch_size, -1, 3), order='C')
        else:
            scmap, locref = _dlc_extract_cnn_output(outputs_np, cfg)
            pose = _dlc_argmax_pose_predict(scmap, locref, cfg["stride"])
            #if outall:
            #    return scmap, locref, pose
            #else:
            return pose

def _dlc_setup_pose_prediction(cfg, locate_on_gpu=False):
    """setting up TF session.

    just copied from DLC1.11; "GPU-inference" part from DLC2.1 (may not work on DLC1.11)."""

    _tf.reset_default_graph()
    inputs = _tf.placeholder(_tf.float32, shape=[cfg["batch_size"]   , None, None, 3])
    net_heads = _pose_net(cfg).test(inputs)

    if locate_on_gpu == False:
        outputs = [net_heads['part_prob']]
        if cfg["location_refinement"] == True:
            outputs.append(net_heads['locref'])
    else:
        if cfg["batch_size"] == 1:
            #assuming batchsize 1 here!
            probs   = _tf.squeeze(net_heads['part_prob'], axis=0)
            locref  = _tf.squeeze(net_heads['locref'], axis=0)
            l_shape = _tf.shape(probs)

            locref  = _tf.reshape(locref, (l_shape[0]*l_shape[1], -1, 2))
            probs   = _tf.reshape(probs , (l_shape[0]*l_shape[1], -1))
            maxloc  = _tf.argmax(probs, axis=0)

            l_shape = _tf.cast(l_shape, _tf.int64)
            n_totalparts = l_shape[2]

        else: # batch_size > 1
            locref  = net_heads['locref']
            probs   = _tf.sigmoid(net_heads['part_prob'])
            l_shape = _tf.shape(probs) # (batchsize, x, y, bodyparts)
            locref  = _tf.reshape(locref, (l_shape[0],l_shape[1],l_shape[2],l_shape[3], 2))

            #turn into x times y time bs * bpts
            locref  = _tf.transpose(locref,[1,2,0,3,4])
            probs   = _tf.transpose(probs,[1,2,0,3])

            l_shape = _tf.shape(probs) # (x, y, batchsize, bodyparts)
            locref  = _tf.reshape(locref, (l_shape[0]*l_shape[1], -1, 2))
            probs   = _tf.reshape(probs , (l_shape[0]*l_shape[1], -1))
            maxloc  = _tf.argmax(probs, axis=0)

            l_shape = _tf.cast(l_shape, _tf.int64)
            n_totalparts = l_shape[2] * l_shape[3]

        loc     = _tf.unravel_index(maxloc, (l_shape[0], l_shape[1])) #tuple of max indices
        maxloc  = _tf.reshape(maxloc, (1, -1))
        joints  = _tf.reshape(_tf.range(0, n_totalparts), (1,-1))
        indices = _tf.transpose(_tf.concat([maxloc, joints] , axis=0))

        # extract corresponding locref x and y as well as probability
        offset  = _tf.gather_nd(locref, indices)
        offset  = _tf.gather(offset, [1,0], axis=1)
        likelihood = _tf.reshape(_tf.gather_nd(probs, indices), (-1,1))

        pose    = cfg.stride * _tf.cast(_tf.transpose(loc), dtype=_tf.float32) + cfg.stride*0.5 + offset*cfg.locref_stdev
        pose    = _tf.concat([pose, likelihood], axis=1)
        outputs = [pose]

    restorer = _tf.train.Saver()
    sess = _tf.Session()
    sess.run(_tf.global_variables_initializer())
    sess.run(_tf.local_variables_initializer())

    # Restore variables from disk.
    restorer.restore(sess, cfg["init_weights"])

    return sess, inputs, outputs, locate_on_gpu

def _dlc_extract_cnn_output(outputs_np, cfg):
    ''' extract locref + scmap from network (juct copied from DLC1.11) '''
    scmap = outputs_np[0]
    scmap = _np.squeeze(scmap)
    locref = None
    if cfg["location_refinement"]:
        locref = _np.squeeze(outputs_np[1])
        shape  = locref.shape
        locref = _np.reshape(locref, (shape[0], shape[1], -1, 2)) * cfg["locref_stdev"]
    if len(scmap.shape)==2: #for single body part!
        scmap = _np.expand_dims(scmap,axis=2)
    return scmap, locref

def _dlc_argmax_pose_predict(scmap, offmat, stride):
    """Combine scoremat and offsets to the final pose. (just copied from DLC1.11)"""
    num_joints = scmap.shape[2]
    pose = []
    for joint_idx in range(num_joints):
        maxloc = _np.unravel_index(_np.argmax(scmap[:, :, joint_idx]),
                                  scmap[:, :, joint_idx].shape)
        offset = _np.array(offmat[maxloc][joint_idx])[::-1]
        pos_f8 = (_np.array(maxloc).astype('float') * stride + 0.5 * stride +
                  offset)
        pose.append(_np.hstack((pos_f8[::-1],
                               [scmap[maxloc][joint_idx]])))
    return _np.array(pose)

def _get_pose_config(cfg, modelfolder, shuffle=1, trainIndex=0):
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

def init_session(cfg, gputouse=None, shuffle=1, trainIndex=0, locate_on_gpu=False):
    if isinstance(cfg, (str, _Path)):
        cfg = _load_config(cfg)
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

    cls = DirectTFSession if cfg["batch_size"] == 1 else BatchTFSession
    return cls(dlc_cfg, *(_dlc_setup_pose_prediction(dlc_cfg, locate_on_gpu=locate_on_gpu)))

from . import writer as _writer
