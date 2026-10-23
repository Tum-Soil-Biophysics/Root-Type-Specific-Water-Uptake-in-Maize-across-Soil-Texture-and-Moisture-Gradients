# Data folder (not on GitHub)

Place the data here, or set the environment variable `RWU_DATA` to another folder with the same structure.

```
Data/
  root_types_lengths.xlsx            root numbers and lengths per sample (Figure 3)
  ColorTemplate.tif                  colour key of the root segmentation (ImageProcessing.py)
  reference_beams/
    ob_.../  *.fits                  open-beam images
    dc_.../  *.fits                  dark-current images
  plants_info/
    to_irrigate.xlsx                 irrigation and transpiration
    plant_morpho.xlsx                plant height and morphology
  night_D2O/
    mean_r.xlsx                      mean root radius per treatment and root type
    <sample>/                        e.g. l_d_1, l_w_2, s_d_3, s_w_4
      Root_Length_Info.xlsx          distance of each root segment from the tip
      D2O_compartment1..3/
        *.fits, seg_*.tif            raw images and segmented roots
        FinalResults/OutPuts.xlsx    written by ImageProcessing.py
  day_D2O/
    mean_r.xlsx
    <sample>/                        <texture>_<moisture>_<id>, e.g. s_w_1, l_d_4
      Root_Length_Info.xlsx            (s = sand, l = loam; w = wet, d = dry)
      D2O_compartment2..4/
        *.fits, seg_*.tif
        FinalResults/OutPuts.xlsx
```

Sample folders in `day_D2O` are discovered automatically from their names.
