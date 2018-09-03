# bsc-integratingcnnsegmentation

### SETUP
1. Copy the `face12` version of the Basel Face Model into the `bfm` folder.
2. \[optional\] Replace the `data_in` and the `PARAMETRIC_rps_files` folder with one of [here](https://github.com/Arneli/image-data)
3. Update the for-loop in `FitWithOcclusions.scala` in line 50 that it runs over all images in `data_in`.

### USAGE
1. Generate a `target` folder by running `sbt assembly` in the repository
2. Replace DUMMY with \[FCN|EGGER|GROTRU|DUMMY\] in `java -cp target/scala-2.12/bsc-integratingCNNSegmentation.jar FitWithOcclusions -d -m bfm/model2017-1_face12_nomouth.h5 -n 1000 -f DUMMY`
3. Update the paths in the command
4. Run the command
