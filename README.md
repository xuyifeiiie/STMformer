### STMformer

---

This repository is for STMformer, a model for system state forecasting tasks of microservices system. 

#### Requirements

---

Install dependencies for the project.

```shell
pip install -r requirements.txt
```

#### Usage

---

1. You can download our raw data from [[Baidu Drive]](https://pan.baidu.com/s/1VtOciqjLYvNZomtLbKbatw?pwd=y5xj)，and place the category-level folder of data into ```./data```. For example, put the normal-15-2h-5s folder in the ```./data```.

2. Then generating and normalizing dataset locally. Horizon steps could be set during dataset generation. You can use the following commands to generate dataset for different prediction steps.

   ```shell
   cd data
   
   # generate dataset
   python ./dataset_generation.py --seq_length_x [history steps] --seq_length_y [prediction steps]
   
   # normalize dataset
   python ./dataset_normalization.py
   ```

3. Don't forget change the settings when you alter the steps. You can also customize experiment settings and select model in the ```./config/TrainTicket_short_term_forecast.conf```.

4. Use following commands to train and evaluate model.  

   ```shell
   cd exp
   python ./exp_forecasting_aiops.py
   ```

   