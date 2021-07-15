# computervisionPanopticToSMPLAuto
Auto convertion of panoptic dataset to SMPL

## Intro
This code was developed on top of the many amazing libraries like VIBE and SMPLX.
The main goal of this work is implementing automated convertion of panoptic dataset to smpl meshes.
We took videos from panoptic dataset and by using their toolbox we extract images from the videos.
Then we feed these pictures which can be from many different camera views to Yolo network to extract human body bounding boxes.
Then using VIBE network we predict many features like human pose and betas(shapes), and save each step separately.
At the end we use this extracted features from many views to compute smpl meshes using smpl network. There are many more detail which can be found in this report.

## Getting Started

Clone the repo
Install the requirements using `virtualenv` or `conda`:
```bash
# pip
source scripts/install_pip.sh

# conda
source scripts/install_conda.sh
```
```bash
source scripts/prepare_data.sh
```
## Running the code 
First step will extract features from camera pictures and save them as raw pickles.
please put images extracted from videos by panoptic toolbox in pics folder
```bash
python step1.py --image_folder_path pics --output_path outpkl
```
Step will prepare our pickle file to be loaded for last phase. This is necessary due to compatiblity issue 
of SMPL net and VIBE net.
```bash
python step3.py --pkl_folder_path outpkl --output_path ashpickle 
```
The last step will take features extracted from previous step, feed to smpl net and get joints and vertices of the 
meshes.
```bash
python step4.py --model-folder models --model-type smpl --batch_pickle ashpickle
```
## RESULT
![output](https://github.com/omidrk/computervisionPanopticToSMPLAuto/blob/main/gifs/output.gif)
![Input random](https://github.com/omidrk/computervisionPanopticToSMPLAuto/blob/main/gifs/input.gif)



## References

- panoptic dataset and toolbox can be found on [link](http://domedb.perception.cs.cmu.edu/)
- VIBE network thanks to [link](https://github.com/mkocabas/VIBE.git)
- SMPLX network thanks to [link](https://github.com/vchoutas/smplx)
- Pretrained HMR and some functions are borrowed from [SPIN](https://github.com/nkolot/SPIN).
- Some functions are borrowed from [Temporal HMR](https://github.com/akanazawa/human_dynamics).
- Some functions are borrowed from [HMR-pytorch](https://github.com/MandyMo/pytorch_HMR).
- Some functions are borrowed from [Kornia](https://github.com/kornia/kornia).
