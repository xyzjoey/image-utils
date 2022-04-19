# Image Utils

helper script to correct image data

```bash
# help
$ python -m tox -e process_png -- -h

usage: process_png.py [-h] [-r] [-g GAMMA] [-d DTYPE] [-c COMPRESS] [steps ...]

file dialog will pop up to ask for input image(s)

positional arguments:
  steps                 run steps in order. available steps: ['revert_gamma', 'correct_gamma', 'unmultiply_alpha', 'crop_transparent']

optional arguments:
  -h, --help            show this help message and exit
  -r, --replace         overwrite result to input path. if not set, output path will be generated
  -g GAMMA, --gamma GAMMA
                        default 2.2
  -d DTYPE, --dtype DTYPE
                        output data type. available: ['uint8', 'uint16']
  -c COMPRESS, --compress COMPRESS
                        compression level. default 9 (smallest file size)
```
```bash
# e.g. remove dark edge of png produced by bleank black ink
$ python -m tox -e process_png -- -d uint8 revert_gamma unmultiply_alpha correct_gamma
```
