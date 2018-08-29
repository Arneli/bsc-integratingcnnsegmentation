# bsc-integratingcnnsegmentation

### SETUP
1. Copy the `face12` version of the Basel Face Model into the `bfm` folder.
2. \[optional\] Replace the `data_in` and the `PARAMETRIC_rps_files` folder with one of [here](https://github.com/Arneli/image-data)
3. Update the for-loop in `FitWithOcclusions.scala` in line 50 that it runs over all images in `data_in`.

### USAGE
1. Replace DUMMY with \[FCN|EGGER|GROTRU|DUMMY\] in `java -cp target/scala-2.12/bsc-integratingCNNSegmentation.jar FitWithOcclusions -d -m bfm/model2017-1_face12_nomouth.h5 -n 1000 -f DUMMY`
2. Update the paths in the command
3. Run the command
4. Copy the outputs into the subfolders of `MATLAB_EVALUATION_SCRIPT`.
5. Create a folder named `image_data` inside the `MATLAB_EVALUATION_SCRIPT` folder.
6. Follow the steps inside the `MATLAB_EVALUATION_SCRIPT/posteriors` folder.
6. Update the for-loop in both MATLAB-Scrips starting with `difference...`, to loop over all the files\(line 12\). Then run it.
7. Update the for-loop in the MATLAB-Script `plot_image_data_average.m` on line 4 to run over all the target images. Then run it to produce the errorplots.
8. Update lene 2 in the MATLAB-Script `segmentation_fit_iteration.m`. then run it to produce a plot which shows the differences of the masks over time.
