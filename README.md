# DA-CRF
* Framework: Bi-lstm-Attention-CRF      
  Gated dynamic Attention
# installation
* python >= 3.5.2    
  tensorflow >= 1.2.0    
  numpy >=1.11.1
  
# Training, developmenting and testing simultaneously    
* python main.py

# Other
* If you want some other parameter settings, you can change them on main.py    
  This project also implements a static attention, see model.py, if you wanna use it, please manually modify on model.py  
  You can change the implemented CoupledInputForgetGateLSTMCell to tensorflow's LSTM_Cell manually in model.py    
  This implementation also suports Chinese, you just feed data like the data on directery "data1" 
