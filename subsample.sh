#EXAMPLE: ./subsample.sh $HOME/dev/lester2/data/scenes/test 1

#Assumes there's a $HOME/DockerVolume/seg4art/data/scenes/tiktok
# with a footage.mp4 file within.


INPUT_DIR=$1
PNG=$2

#mkdir -p $OUTPUT_DIR

cd $INPUT_DIR

#To subsample to less than 24 FPS:
#mkdir images
#if [ "$PNG" == 1 ]; then
#    ffmpeg -i $INPUT_DIR/footage.mp4 -r 5 $"images/$no_ext%03d.png"
#else
#    ffmpeg -i $INPUT_DIR/footage.mp4 -r 5 $"images/$no_ext%03d.jpg"
#fi 

#To just pick 24 FPS
mkdir imagesFull
if [ "$PNG" == 1 ]; then
    ffmpeg -i $INPUT_DIR/footage.mp4 -r 24 $"imagesFull/$no_ext%03d.png"
else
    ffmpeg -i $INPUT_DIR/footage.mp4 -r 24 $"imagesFull/$no_ext%03d.jpg"
fi 

#echo "INPUT_DIR="$INPUT_DIR
#echo "PNG="$PNG
