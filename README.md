# To run this code
1. Add a 'data_in' and a 'PARAMETRIC_rps_files' folder of the 'image-data' repository
2. Create four folders named: 'data_out','segmentations','rps' and a 'fits'
3. Add the basel face model (face12) to the 'bfm' folder
4. Run 'sbt assembly' in the repository folder
5. Run 'java -cp target/scala-2.12/bsc-integratingCNNSegmentation.jar FitWithOcclusions -d -m bfm/model2017-1_face12_nomouth.h5 -n 1000 -f <Mask>' and replace <Mask> with EGGER,FCN,DUMMY or GROTRU

