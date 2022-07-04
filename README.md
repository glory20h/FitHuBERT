# FitHuBERT
This repository is for supplementing the paper, ["FitHuBERT: Going Thinner and Deeper for Knowledge Distillation of Speech Self-Supervised Learning"](https://arxiv.org/abs/2207.00555), INTERSPEECH 2022.

## Distillation
1. Download the model checkpoint to perform knowledge distillation (e.g. [HuBERT Base](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert)):

2. Download the [LibriSpeech](https://www.openslr.org/12) dataset.

Modify the configuration file in `/data/conf/`. The configuration file `fithubert.yaml` contains all the settings for reproducing FitHuBERT. Set the path to the teacher model checkpoint at `teacher_model`, and the root path to the LibriSpeech dataset at `libri_root`. 

Then, run the following command:
```
python train.py --config ./data/conf/fithubert.yaml
```

After training, the model checkpoints and the corresponding configuration file will be created at `/results/pretrain/`.

## Using the model for downstream tasks
1. Download and install the [S3PRL toolkit](https://github.com/s3prl/s3prl).

2. Copy the `fithubert` folder into `s3prl/upstream/`.

3. Run the following the command to use the FitHuBERT model for automatic speech recognition(ASR).

```
python run_downstream.py -m train -n FitHuBERT-ASR -u fithubert -d asr -k <path to .ckpt file> -g <path to .yaml file>
```

Refer to the [SUPERB docs](https://github.com/s3prl/s3prl/blob/master/s3prl/downstream/docs/superb.md) for more information on usage details and data preparation.

## Citation
To cite our paper:
```
@article{lee2022fithubert,
  title={FitHuBERT: Going Thinner and Deeper for Knowledge Distillation of Speech Self-Supervised Learning},
  author={Lee, Yeonghyeon and Jang, Kangwook and Goo, Jahyun and Jung, Youngmoon and Kim, Hoirin},
  journal={arXiv preprint arXiv:2207.00555},
  year={2022}
}
```
