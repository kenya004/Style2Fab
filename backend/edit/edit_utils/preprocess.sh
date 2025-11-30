clear
eval "$(conda shell.bash hook)"
conda activate fa3ds
screen -d -m -S "preprocess_screen" -L -Logfile C:\\Users\\mrkab\\git\\Style2Fab/backend/edit/edit_utils/preprocess.log python3 C:\\Users\\mrkab\\git\\Style2Fab/backend/edit/edit_utils/preprocess.py
screen -x