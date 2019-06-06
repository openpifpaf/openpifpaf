# Performance Studies

Might be out of date.


# Profiling Decoder

![a COCO image](coco/000000081988.jpg.skeleton.png)

Run predict with the `--profile` option:

```sh
python3 -m openpifpaf.predict --checkpoint resnet152 \
  docs/coco/000000081988.jpg --show --profile --debug
```

This will write a stats table to the terminal and also produce a `decoder.prof` file.
You can use flameprof (`pip install flameprof`) to get a flame graph with
`flameprof decoder.prof > docs/coco/000000081988.jpg.decoder_flame.svg`:

![flame graph for decoder on a COCO image](coco/000000081988.jpg.decoder_flame.svg)


### Large image

e.g., from NuScenes:

![a NuScenes image](nuscenes/test.jpg.skeleton.png)

```
python3 -m openpifpaf.predict --checkpoint resnet152 \
  docs/nuscenes/test.jpg --show --profile --debug
```

Then create the flame graph with
`flameprof decoder.prof > docs/nuscenes/test.jpg.decoder_flame.svg` to produce:

![flame graph for decoder on a NuScenes image](nuscenes/test.jpg.decoder_flame.svg)


### Crowded image

![crowded image](crowd.png.skeleton.png)

```
python3 -m openpifpaf.predict --checkpoint resnet152 \
  docs/crowd.png --show --profile --debug
```

Then create the flame graph with
`flameprof decoder.prof > docs/crowd.png.decoder_flame.svg` to produce:

![flame graph for decoder on a crowded image](crowd.png.decoder_flame.svg)


### Low quality crowded scene

![Mochit station image](mochit_station_example.jpg.skeleton.png)

```
python3 -m openpifpaf.predict --checkpoint resnet152 \
  docs/mochit_station_example.jpg --show --profile --debug
```

Then create the flame graph with
`flameprof decoder.prof > docs/mochit_station_example.jpg.decoder_flame.svg` to produce:

![flame graph for decoder on a Mochit station image](mochit_station_example.jpg.decoder_flame.svg)
