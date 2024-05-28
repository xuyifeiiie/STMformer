### STMformer

---

This repository is for STMformer, a model for system state forecasting tasks of microservices system. 

#### Requirements

---

Install dependencies for the project.

```python
pip install -r requirements.txt
```

#### Usage

---

1. You can download our processed dataset from [[Baidu Drive]]() and place the processed dataset into folder ```./data```

2. Use following commands to train and evaluate model.  

   ```shell
   cd exp
   python ./exp_forecasting_aiops.py
   ```

3. You can customize experiment settings and alter model in the ```./config/TrainTicket_short_term_forecast.conf```

4. Default horizon steps is 16. You can use the following commands to generate dataset for different prediction steps.

   ```shell
   cd data
   
   # generate dataset
   python ./dataset_generation.py --seq_length_x [history steps] --seq_length_y [prediction steps]
   
   # normalize dataset
   python ./dataset_normalization.py
   ```

   