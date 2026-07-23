# In-car noise collection protocol

This protocol supports noise augmentation for the InCar-ASR system. The project target is at least 10 representative vehicle-noise categories and evaluation at 5–15 dB SNR.

## Safety and privacy

- The driver must never operate the recorder. A passenger should start and stop collection, or the device should be configured before driving.
- Do not create unsafe traffic situations merely to obtain a sound.
- Obtain consent before recording occupants. Avoid collecting identifiable conversation.
- Do not store names, exact locations, vehicle plates, phone calls, or other personal information in filenames or metadata.
- Check the license of every external recording and retain its source URL and license in metadata.

## Required categories

The canonical labels are defined in `configs/audio.yaml`. They cover engine idle and acceleration, multiple road/tire surfaces, wind, air conditioning, rain, traffic/horns, cabin media or speech, and open-window conditions.

Collect all categories before adding fine-grained labels. If one recording contains several dominant sources, choose the dominant category and explain the secondary sources in `notes`.

## Scene coverage

For each category, vary conditions where safe and practical:

- vehicle state: parked, idling, accelerating, steady speed;
- speed band: 0, 1–30, 31–60, and above 60 km/h;
- road surface: asphalt, concrete, rough or wet road;
- weather: dry, rain, or strong wind;
- windows: closed, partly open, or open;
- microphone position: driver area, center console, or rear seat.

Aim for several independent recording sessions per condition. Long recordings are useful, but splits from the same recording are not independent and must stay in one train/validation/test partition.

## Recording procedure

1. Use WAV, 16 kHz, mono, PCM 16-bit unless the input device requires a higher native rate.
2. Place and secure the microphone; record its position in metadata.
3. Record a short test and check that the peak does not clip and the signal is not extremely quiet.
4. Record at least 30 seconds for a scene, without changing the scene label midway.
5. Stop safely, confirm the generated metadata, and listen to a short excerpt.
6. Preserve the raw file. Correct only its metadata if needed.

Example:

```bash
python modules/01_dataset_building/scripts/audio/collect_audio.py \
  --category road_asphalt \
  --duration 60 \
  --vehicle-state steady_speed \
  --speed-kmh 50 \
  --road-surface asphalt \
  --weather dry \
  --window-state closed \
  --microphone-position center_console
```

## Acceptance checks

Before handing the dataset to the model-training member:

- at least 10 categories are represented;
- files and metadata agree;
- recordings have no obvious corruption or severe clipping;
- each external file has a traceable source and license;
- identifiable speech is removed or access-restricted;
- no source recording appears in more than one dataset split;
- category counts and total duration are reported.

Use `validate_dataset.py` to automate the format and metadata checks, then perform random listening checks because signal statistics cannot detect every content problem.
