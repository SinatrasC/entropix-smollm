# entropix-smollm
## smolLM with Entropix sampler on pytorch

This notebook has Entropix Sampler implementation by Sinatras [@myainotez](https://x.com/myainotez)

Special thanks to [@Dorialexander](https://x.com/Dorialexander) , [@citizenhicks](https://x.com/citizenhicks) and original entropix implementation maintainer [@_xjdr](https://x.com/_xjdr) for making this possible

There are 2 different SamplerConfigs included, both are experimental one of them is leaning into adaptive state as before and another one is forcing model to use different sampler states to let this agreement is removed from conditions. You can customize your sampler parameters using 3D chart easily even you can identify tokens by hovering them. You can also setup your EntropixConfig to utilize or remove some parts of the sampler for research purposes.

To disable charts set ```debug=False``` in EntropixModel class

To enable export of varentropy&entropy stats for attention and logits remove comment on ```export_data``` function

<img width="650" alt="smolLMxEntropix" src="https://github.com/user-attachments/assets/a7b1834b-4cd3-490b-983d-2479dc53c9e2">

## 3D Entropy&Varentropy Chart
New 3D chart lets user see how their response is formed, includes varentropy&entropy stats for attention and logits

![3D1](https://github.com/user-attachments/assets/4ecd74ec-6377-4961-8262-82286df8c765)
![3DGIF](https://github.com/user-attachments/assets/8c044476-bbe9-4849-b28a-e28b6f192418)

There is also threshold visualization to let users see how their responses fill 3D space with desired SamplerConfig, users can use this function with buttons on top of the chart

![3D2](https://github.com/user-attachments/assets/bf823633-e4eb-404c-be54-8f9ef9500565)

## Samples
![Q1](https://github.com/user-attachments/assets/adb455ef-d3bb-41b5-af14-815e048fded8)

![Q2](https://github.com/user-attachments/assets/062eaf0d-b0e1-4a21-98fe-b85adc8450e8)

Original entropix implementation : @xjdr-alt/entropix

## Install

Clone the repo

```sh
git clone git@github.com:SinatrasC/entropix-smollm.git

cd entropix-smollm
```

## With `rye`
Install Rye [here](https://rye.astral.sh/) if you haven't already, and then:

```sh
rye sync
```

Run Jupyter

```sh
rye run jupyter-notebook smollm_entropix_torch.ipynb
```


### With `uv`
Install uv [here](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already (Rye installs it by default), and then:

```sh
uv venv --python 3.12

source .venv/bin/activate

uv pip install --project pyproject.toml .
```

and then:

```sh
jupyter-notebook smollm_entropix_torch.ipynb 
```
