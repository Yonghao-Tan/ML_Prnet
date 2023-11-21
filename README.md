# Partial Registration Network
This is a repo fetch originally from https://github.com/yfeng95/PRNet, as our `COMP 5212 - Machine Learning` project.

## Prerequisites 
The `environment.yaml` file is depreciated, we recommende to set up the env follow the version requirement below: 
|Packgage | Version| 
|-| -| 
|PyTorch| 1.0.1 (must equal, PyTorch 1.1 has a svd bug, which will crash the training)|
|scipy | 1.1.0 (must equal) | 
|open3d|0.17.0 (for visualization only)|
|numpy, h5py, tqdm, sklearn| not specified|


`conda activate prnet`

## Training

### exp1 modelnet40

`python main.py --exp_name=exp1`

### exp2 modelnet40 unseen
`
python main.py --exp_name=exp2 --unseen=True`

## exp3 modelnet40 gaussian noise

`python main.py --exp_name=exp3 --gaussian_noise=True`

## Evaluation
`python main.py --exp_name exp2 --unseen True --eval`

You can try evaluation first.

## Visualization 
The visualization is implemented on [Stanfor Bunny](https://graphics.stanford.edu/data/3Dscanrep/).

`python3 main.py --exp_name exp2 --eval --dataset stfbunny --vis 1`

## Citation
Please cite this paper if you want to use it in your work,

	@InProceedings{Wang_2019_NeurIPS,
	  title={PRNet: Self-Supervised Learning for Partial-to-Partial Registration},
	  author={Wang, Yue and Solomon, Justin M.},
	  booktitle = {33rd Conference on Neural Information Processing Systems (To appear)},
	  year={2019}
	}

## Code Reference

Code reference: Deep Closest Point

## License
MIT License
