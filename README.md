# TDAR
This repo is for source code of paper "Topology-Driven Attribute Recovery for Attribute Missing Graph Learning in Social Internet of Things". This code is referenced from [SVGA](https://github.com/snudatalab/SVGA) and [SAT](https://github.com/xuChenSJTU/SAT-master-online), thanks to the author's contribution.

## Environment Settings
> python==3.11.4 \
> scipy==1.10.1 \
> torch==2.0.1 \
> torch-geometric==2.3.1 \
> numpy==1.24.3 \
> scikit_learn==1.3.0 \
GPU: GeForce RTX 4090 
## Dataset
The data obtained from [link](https://github.com/xuChenSJTU/SAT-master-online) is approximate.


## Usage
You can use the following command to run our model.
> python main.py --data cora --lr 1e-3 --dropout 0.8 --layers 2 --epochs 2000 --conv lin
> 
> python main.py --data citeseer --lr 1e-3 --dropout 0.8 --layers 2 --epochs 2000 --conv lin
> 
> python main.py --data amac --lr 1e-2 --dropout 0.2 --layers 1 --epochs 400 --conv gcn
> 
> python main.py --data amap --lr 1e-2 --dropout 0.2 --layers 1 --epochs 400 --conv gcn

Note: It is undoubtedly extremely difficult to achieve the best results for all downstream tasks (feature reconstruction, node classification, node clustering) with unified hyperparameters ($\lambda_1$, $\lambda_2$, $l$, $\epsilon$ and $\alpha$). Therefore, different downstream tasks have different requirements for hyperparameters. If you want to reproduce the results in our paper, please refer to Section IV.G.

## Contcat
This code has not been thoroughly verified and is only for learning and communication purposes. Please feel free to raise any questions or suggest better solutions by contacting limengran1998@163.com.
