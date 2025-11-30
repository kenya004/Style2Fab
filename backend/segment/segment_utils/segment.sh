clear
eval "$(conda shell.bash hook)"
conda activate fa3ds
screen -d -m -S "segmentations_screen" -L -Logfile segmentation.log python3 C:\\Users\\mrkab\\git\\Style2Fab/backend/segment/segment_utils/mesh_editor.py
screen -x