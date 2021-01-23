# python-dlclib

A wrapper of DeepLabCut for its ease of use.

`dlclib` is supposed to work with almost all single-camera setups of all DLC versions.

## Installation

It is better building your DeepLabCut environment first, before running:

```bash
$ pip install dlclib
```

## Classes

### `dlclib.estimate.TFSession`

A thin wrapper over the DeepLabCut inference network based on TensorFlow.

- `session = TFSession.from_config(cfg)` will return a `TFSession` object
  using the specified `config.yaml` file of the DeepLabCut project.
- `session.get_pose(image)` will return a `numpy.ndarray` corresponding to the estimation.
  for a batch-processing network, use a list of images, or a 4-D uint8 array
  whose axis 0 correspond to indexes of images.
- `session.open_writer(path)` opens a CSVWriter at the specified path (see below).

### `dlclib.posture.PostureMapper`

A utility class that maps the output of `TFSession` to a dictionary.
Note that the current implementation:

- does not care anything about conversion speed.
- currently only supports non-batch models.

- `mapper = PostureMapper.from_config(cfg)` will return a `PostureMapper` object
  using the specified `config.yaml` file of the DeepLabCut project.
- `posedict = mapper.map(pose)` receives the output of `TFSession.get_pose()`,
  and returns a dictionary. `posedict[part]` is a named tuple that has `x`, `y` and `prob` attributes.

### `dlclib.writer.CSVWriter`

A simple class for writing the estimation into a CSV file.
This class _does_ support batch-processing networks.

- `writer = session.open_writer(path)` from a `TFSession` object initializes the writer
  associated with the inference session.
- For every video frame, you can call `writer.push(frame)`. It automatically calls
  `TFSession.get_pose()` internally, and writes its outputs into the file.
- Calling `writer.flush()` will flush the pended video frames (in case of batch modes;
  it does not wait for more frames and runs `get_pose()` to writes the estimation).
- Calling `writer.close()` will flush its internal buffer, close the file and invalidates itself.
  You can keep using the originating `TFSession` object.

This class supports the context-manager model, so you can write:

```python
with session.open_writer(path) as out:
    for frame in video_frames:
        out.push(frame)
```

## License

2020-2021 Keisuke Sehara, the MIT License
